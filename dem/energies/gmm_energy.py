from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from fab.target_distributions import gmm
from fab.utils.plotting import plot_contours, plot_marginal_pair
from lightning.pytorch.loggers import WandbLogger

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.models.components.replay_buffer import ReplayBuffer
from dem.utils.logging_utils import fig_to_image

import torch.nn as nn
import torch.nn.functional as f
from typing import Optional, Dict
from fab.types_ import LogProbFunc
from fab.target_distributions.base import TargetDistribution
from fab.utils.numerical import MC_estimate_true_expectation, importance_weighted_expectation, effective_sample_size_over_p

def setup_quadratic_function(x: torch.Tensor, seed: int = 0):
    # Useful for porting this problem to non torch libraries.
    torch.random.manual_seed(seed)
    # example function that we may want to calculate expectations over
    x_shift = 2 * torch.randn(x.shape[-1]).to(x.device)
    A = 2 * torch.rand((x.shape[-1], x.shape[-1])).to(x.device)
    b = torch.rand(x.shape[-1]).to(x.device)
    torch.seed()  # set back to random number
    if x.dtype == torch.float64:
        return x_shift.double(), A.double(), b.double()
    else:
        assert x.dtype == torch.float32
        return x_shift, A, b


def quadratic_function(x: torch.Tensor, seed: int = 0):
    x_shift, A, b = setup_quadratic_function(x, seed)
    x = x + x_shift
    return torch.einsum("bi,ij,bj->b", x, A, x) + torch.einsum("i,bi->b", b, x)

class GMMGPU(nn.Module, TargetDistribution):
    def __init__(self, dim, n_mixes, loc_scaling, log_var_scaling=0.1, seed=0,
                 n_test_set_samples=1000, use_gpu=True,
                 true_expectation_estimation_n_samples=int(1e7)):
        super(GMMGPU, self).__init__()
        self.seed = seed
        self.n_mixes = n_mixes
        self.dim = dim
        self.n_test_set_samples = n_test_set_samples

        device = "cuda"
        mean = (torch.rand((n_mixes, dim), device=device) - 0.5)*2 * loc_scaling
        log_var = torch.ones((n_mixes, dim), device=device) * log_var_scaling
        self.register_buffer("cat_probs", torch.ones(n_mixes).to(device))
        self.register_buffer("locs", mean)
        self.register_buffer("scale_trils", torch.diag_embed(f.softplus(log_var)))
        self.expectation_function = quadratic_function
        self.register_buffer("true_expectation", MC_estimate_true_expectation(self,
                                                             self.expectation_function,
                                                             true_expectation_estimation_n_samples
                                                                              ))
        self.device = device 
        self.to(self.device)

    def to(self, device):
        if device == "cuda":
            if torch.cuda.is_available():
                self.cuda()
        else:
            self.cpu()

    @property
    def distribution(self):
        mix = torch.distributions.Categorical(self.cat_probs)
        com = torch.distributions.MultivariateNormal(self.locs,
                                                     scale_tril=self.scale_trils,
                                                     validate_args=False)
        return torch.distributions.MixtureSameFamily(mixture_distribution=mix,
                                                     component_distribution=com,
                                                     validate_args=False)

    @property
    def test_set(self) -> torch.Tensor:
        return self.sample((self.n_test_set_samples, ))

    def log_prob(self, x: torch.Tensor):
        # print(x.device)
        log_prob = self.distribution.log_prob(x)
        # Very low probability samples can cause issues (we turn off validate_args of the
        # distribution object which typically raises an expection related to this.
        # We manually decrease the distributions log prob to prevent them having an effect on
        # the loss/buffer.
        # Use torch.where instead of boolean indexing for vmap compatibility
        log_prob = torch.where(
            log_prob < -1e4,
            log_prob - torch.tensor(float("inf"), device=log_prob.device, dtype=log_prob.dtype),
            log_prob
        )
        return log_prob

    def sample(self, shape=(1,)):
        return self.distribution.sample(shape)

    def evaluate_expectation(self, samples, log_w):
        expectation = importance_weighted_expectation(self.expectation_function,
                                                         samples, log_w)
        true_expectation = self.true_expectation.to(expectation.device)
        bias_normed = (expectation - true_expectation) / true_expectation
        return bias_normed

    def performance_metrics(self, samples: torch.Tensor, log_w: torch.Tensor,
                            log_q_fn: Optional[LogProbFunc] = None,
                            batch_size: Optional[int] = None) -> Dict:
        bias_normed = self.evaluate_expectation(samples, log_w)
        bias_no_correction = self.evaluate_expectation(samples, torch.ones_like(log_w))
        if log_q_fn:
            log_q_test = log_q_fn(self.test_set)
            log_p_test = self.log_prob(self.test_set)
            test_mean_log_prob = torch.mean(log_q_test)
            kl_forward = torch.mean(log_p_test - log_q_test)
            ess_over_p = effective_sample_size_over_p(log_p_test - log_q_test)
            summary_dict = {
                "test_set_mean_log_prob": test_mean_log_prob.cpu().item(),
                "bias_normed": torch.abs(bias_normed).cpu().item(),
                "bias_no_correction": torch.abs(bias_no_correction).cpu().item(),
                "ess_over_p": ess_over_p.detach().cpu().item(),
                "kl_forward": kl_forward.detach().cpu().item()
                            }
        else:
            summary_dict = {"bias_normed": bias_normed.cpu().item(),
                            "bias_no_correction": torch.abs(bias_no_correction).cpu().item()}
        return summary_dict
    
class GMM(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality=2,
        n_mixes=40,
        loc_scaling=40,
        log_var_scaling=1.0,
        device="cpu",
        true_expectation_estimation_n_samples=int(1e5),
        plotting_buffer_sample_size=512,
        plot_samples_epoch_period=5,
        should_unnormalize=False,
        data_normalization_factor=50,
        train_set_size=100000,
        test_set_size=2000,
        val_set_size=2000,
        data_path_train=None,
    ):
        # use_gpu = device != "cpu"
        use_gpu = True
        torch.manual_seed(0)  # seed of 0 for GMM problem
        self.gmm = GMMGPU(
            dim=dimensionality,
            n_mixes=n_mixes,
            loc_scaling=loc_scaling,
            log_var_scaling=log_var_scaling,
            use_gpu=use_gpu,
            true_expectation_estimation_n_samples=true_expectation_estimation_n_samples,
        )

        self.curr_epoch = 0
        self.device = device
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period

        self.should_unnormalize = should_unnormalize
        self.data_normalization_factor = data_normalization_factor

        self.train_set_size = train_set_size
        self.test_set_size = test_set_size
        self.val_set_size = val_set_size

        self.data_path_train = data_path_train

        self.name = "gmm"

        super().__init__(
            dimensionality=dimensionality,
            normalization_min=-data_normalization_factor,
            normalization_max=data_normalization_factor,
        )

    def setup_test_set(self):
        # test_sample = self.gmm.sample((self.test_set_size,))
        # return test_sample
        return self.gmm.test_set

    def setup_train_set(self):
        if self.data_path_train is None:
            train_samples = self.normalize(self.gmm.sample((self.train_set_size,)))

        else:
            # Assume the samples we are loading from disk are already normalized.
            # This breaks if they are not.

            if self.data_path_train.endswith(".pt"):
                data = torch.load(self.data_path_train).cpu().numpy()
            else:
                data = np.load(self.data_path_train, allow_pickle=True)

            data = torch.tensor(data, device=self.device)

        return train_samples

    def setup_val_set(self):
        val_samples = self.gmm.sample((self.val_set_size,))
        return val_samples

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        if self.should_unnormalize:
            samples = self.unnormalize(samples)

        return self.gmm.log_prob(samples)

    @property
    def dimensionality(self):
        return 2

    def log_on_epoch_end(
        self,
        latest_samples: torch.Tensor,
        latest_energies: torch.Tensor,
        wandb_logger: WandbLogger,
        unprioritized_buffer_samples=None,
        cfm_samples=None,
        replay_buffer=None,
        prefix: str = "",
    ) -> None:
        if wandb_logger is None:
            return

        if len(prefix) > 0 and prefix[-1] != "/":
            prefix += "/"

        if self.curr_epoch % self.plot_samples_epoch_period == 0:
            if self.should_unnormalize:
                # Don't unnormalize CFM samples since they're in the
                # unnormalized space
                if latest_samples is not None:
                    latest_samples = self.unnormalize(latest_samples)

                if unprioritized_buffer_samples is not None:
                    unprioritized_buffer_samples = self.unnormalize(unprioritized_buffer_samples)

            if unprioritized_buffer_samples is not None:
                buffer_samples, _, _ = replay_buffer.sample(self.plotting_buffer_sample_size)
                if self.should_unnormalize:
                    buffer_samples = self.unnormalize(buffer_samples)

                samples_fig = self.get_dataset_fig(buffer_samples, latest_samples)

                wandb_logger.log_image(f"{prefix}unprioritized_buffer_samples", [samples_fig])

            if cfm_samples is not None:
                cfm_samples_fig = self.get_dataset_fig(unprioritized_buffer_samples, cfm_samples)

                wandb_logger.log_image(f"{prefix}cfm_generated_samples", [cfm_samples_fig])

            if latest_samples is not None:
                fig, ax = plt.subplots()
                ax.scatter(*latest_samples.detach().cpu().T)

                wandb_logger.log_image(f"{prefix}generated_samples_scatter", [fig_to_image(fig)])
                img = self.get_single_dataset_fig(latest_samples, "dem_generated_samples")
                wandb_logger.log_image(f"{prefix}generated_samples", [img])

            plt.close()

        self.curr_epoch += 1

    def log_samples(
        self,
        samples: torch.Tensor,
        wandb_logger: WandbLogger,
        name: str = "",
        should_unnormalize: bool = False,
    ) -> None:
        if wandb_logger is None:
            return

        if self.should_unnormalize and should_unnormalize:
            samples = self.unnormalize(samples)
        samples_fig = self.get_single_dataset_fig(samples, name)
        wandb_logger.log_image(f"{name}", [samples_fig])

    def get_single_dataset_fig(self, samples, name, plotting_bounds=(-1.4 * 40, 1.4 * 40)):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        self.gmm.to("cpu")
        plot_contours(
            self.gmm.log_prob,
            bounds=plotting_bounds,
            ax=ax,
            n_contour_levels=50,
            grid_width_n_points=200,
        )

        plot_marginal_pair(samples, ax=ax, bounds=plotting_bounds)
        ax.set_title(f"{name}")

        self.gmm.to(self.device)

        return fig_to_image(fig)

    def get_dataset_fig(self, samples, gen_samples=None, plotting_bounds=(-1.4 * 40, 1.4 * 40)):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        self.gmm.to("cpu")
        plot_contours(
            self.gmm.log_prob,
            bounds=plotting_bounds,
            ax=axs[0],
            n_contour_levels=50,
            grid_width_n_points=200,
        )

        # plot dataset samples
        plot_marginal_pair(samples, ax=axs[0], bounds=plotting_bounds)
        axs[0].set_title("Buffer")

        if gen_samples is not None:
            plot_contours(
                self.gmm.log_prob,
                bounds=plotting_bounds,
                ax=axs[1],
                n_contour_levels=50,
                grid_width_n_points=200,
            )
            # plot generated samples
            plot_marginal_pair(gen_samples, ax=axs[1], bounds=plotting_bounds)
            axs[1].set_title("Generated samples")

        # delete subplot
        else:
            fig.delaxes(axs[1])

        self.gmm.to(self.device)

        return fig_to_image(fig)
