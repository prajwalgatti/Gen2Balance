def partial_freeze(model, n_partial=0.5, print_details=False):
    # Total number of blocks
    total_blocks = len(model.blocks)

    # Assertions
    assert n_partial >= 0.5, "n_partial must be at least 0.5"
    assert n_partial.is_integer() or n_partial == 0.5, "n_partial must be an integer or 0.5"
    assert n_partial <= total_blocks, f"n_partial must be less than or equal to the total number of blocks ({total_blocks})"
    # assert n_partial != 0.5, "Implement/set code to use correct hyperparameters for linear probing (e.g., use SGD, no weight decay, etc.)"

    # freeze the embedding layer
    for param in model.patch_embed.parameters():
        param.requires_grad = False

    # Calculate the first block to train
    if n_partial == 0.5:
        first_train_block = int(total_blocks)
    else:
        first_train_block = int(total_blocks - n_partial)

    # Freeze layers up to the first block to train
    for block_idx in range(first_train_block):
        for param in model.blocks[block_idx].parameters():
            param.requires_grad = False

    # Optionally, print which layers are frozen/unfrozen
    if print_details:
        for name, param in model.named_parameters():
            print(f"{name} is {'not ' if param.requires_grad else ''}frozen")

    return model
