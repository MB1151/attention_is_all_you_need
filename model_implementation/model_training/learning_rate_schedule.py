# This file implements the learning rate schedule for the transformer model. The learning rate is increased 
# linearly for the first warmup_steps, and then decreased exponentially for the rest of the training steps. 
# Refer to 'step_18_learning_rates.ipynb' (link to the notebook) for a detailed explanation of how this 
# learning rate schedule works.

from typing import Optional

def rate(step: int, d_model: int, warmup_steps: int, factor: Optional[float] = 1.0) -> float:
    """This functions implements the above mentioned learning rate schedule. The learning rate is increased linearly
       for the first warmup_steps, and then decreased exponentially for the rest of the training steps. step 
       corresponds to 'epoch' number in the adam_optimizer functionality in pytorch.

    Args:
        step (int): current epoch number in the training loop. starts from 0.
        d_model (int): size of the vectors in the model. This is 512 in the original transformer model.
        warmup_steps (int): number of steps to increase the learning rate linearly.
        factor (Optional[float], optional): factor to scale the learning rate. Defaults to 1.0.

    Returns:
        float: returns the learning rate by which the initial learning rate should be scaled.
    """
    if step == 0:
        step = 1
    return factor * (d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5)))