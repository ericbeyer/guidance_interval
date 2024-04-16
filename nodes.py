from copy import deepcopy
import comfy.samplers
import torch
import math

original_sampling_function = deepcopy(comfy.samplers.sampling_function)

def sampling_function_patched(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
    sigma = timestep[0]
    sigma_min = model_options.get("sigma_min", 0.28)
    sigma_max = model_options.get("sigma_max", 5.42)
    use_guidance_interval = model_options.get("use_guidance_interval", False)
    if use_guidance_interval:
        guidance_weight = model_options.get("guidance_weight", 16.0)

        conds = [cond, uncond]
        if sigma_min < sigma <= sigma_max:
            # Apply guidance within the interval
            conds = [cond, uncond]
            out = comfy.samplers.calc_cond_batch(model, conds, x, timestep, model_options)
            cond_pred, uncond_pred = out
            cfg_result = uncond_pred + guidance_weight * (cond_pred - uncond_pred)
        else:
            # # Disable guidance outside the interval
            conds = [None, uncond]
            out = comfy.samplers.calc_cond_batch(model, conds, x, timestep, model_options)
            cond_pred, uncond_pred = out
            cfg_result = uncond_pred


        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred,
                    "sigma": timestep, "model_options": model_options, "input": x}
            cfg_result = fn(args)

        return cfg_result

    else:
        return original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)

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
        m.model_options["use_guidance_interval"] = True
        m.model_options["guidance_weight"] = guidance_weight
        m.model_options["sigma_min"] = sigma_min
        m.model_options["sigma_max"] = sigma_max
        return (m,)