from .deformable_detr import build


def build_model(args, device):
    return build(args, device)