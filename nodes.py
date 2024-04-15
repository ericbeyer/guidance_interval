from copy import deepcopy
import comfy.samplers
import torch
import math


def sampling_function_patched(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
    sigma = timestep[0]
    sigma_min = model_options.get("sigma_min", 0.28)
    sigma_max = model_options.get("sigma_max", 5.42)

    if sigma_min < sigma <= sigma_max:
        # Apply guidance within the interval
        conds = [cond, uncond]
        out = comfy.samplers.calc_cond_batch(model, conds, x, timestep, model_options)
        cond_pred, uncond_pred = out
        denoised = uncond_pred + cond_scale * (cond_pred - uncond_pred)
    else:
        # Disable guidance outside the interval
        out = comfy.samplers.calc_cond_batch(model, [cond], x, timestep, model_options)
        denoised = out[0]

    # # Clamp the denoised output
    # denoised = torch.clamp(denoised, -1.0, 1.0)

    return denoised

class GuidanceInterval:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "guidance_weight": ("FLOAT", {"default": 16.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                                "sigma_min": ("FLOAT", {"default": 0.28, "min": 0.0, "max": 1.0, "step": 0.01}),
                                "sigma_max": ("FLOAT", {"default": 5.42, "min": 0.0, "max": 80.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches"

    def patch(self, model, guidance_weight, sigma_min, sigma_max):
        comfy.samplers.sampling_function = sampling_function_patched
        
        m = model.clone()
        m.model_options["guidance_weight"] = guidance_weight
        m.model_options["sigma_min"] = sigma_min
        m.model_options["sigma_max"] = sigma_max
        return (m,)