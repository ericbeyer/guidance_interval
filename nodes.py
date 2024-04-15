from copy import deepcopy
import comfy.samplers
import torch
import math

original_sampling_function = deepcopy(comfy.samplers.sampling_function)
minimum_sigma_to_disable_uncond = 1
no_uncond_at_all = False


class GuidanceInterval:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "guidance_weight": ("FLOAT", {"default": 16.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                                "sigma_min": ("FLOAT", {"default": 0.28, "min": 0.0, "max": 1.0, "step": 0.01}),  
                                "sigma_max": ("FLOAT", {"default": 5.42, "min": 0.0, "max": 10.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches"

    def patch(self, model, guidance_weight, sigma_min, sigma_max):
        def guidance_function(args):
            sigma = args["sigma"][0]
            cond_denoised = args["cond_denoised"]
            uncond_denoised = args["uncond_denoised"]
            x_in = args["input"]

            if sigma_min < sigma <= sigma_max:
                # Apply guidance in interval
                denoised = uncond_denoised + guidance_weight * (cond_denoised - uncond_denoised)
            else:
                # Disable guidance outside interval
                denoised = uncond_denoised

            # Clip the denoised image to a valid range
            denoised = torch.clamp(denoised, -1.0, 1.0)

            # Compute the final output
            return x_in - denoised

        m = model.clone()
        m.set_model_sampler_cfg_function(guidance_function)
        return (m,)