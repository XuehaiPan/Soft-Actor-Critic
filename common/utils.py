def sync_params(src_net, dst_net, soft_tau=1.0):
    if soft_tau == 1.0:
        dst_net.load_state_dict(src_net.state_dict())
    else:
        for src_param, dst_param in zip(src_net.parameters(), dst_net.parameters()):
            dst_param.data.copy_(dst_param.data * (1.0 - soft_tau) + src_param.data * soft_tau)
