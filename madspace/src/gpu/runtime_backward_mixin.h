// This file was automatically generated based on instruction_set.yaml
// Do not modify its content directly

case 0:
    backward_op_stack(instr, locals, local_grads, device);
    break;
case 1:
    backward_op_unstack(instr, locals, local_grads, device);
    break;
case 3:
    backward_op_pop(instr, locals, local_grads, device);
    break;
case 4:
    backward_op_batch_cat(instr, locals, local_grads, device);
    break;
case 5:
    backward_op_batch_split(instr, locals, local_grads, device);
    break;
case 6:
    backward_op_cat(instr, locals, local_grads, device);
    break;
case 10:
    backward_op_squeeze(instr, locals, local_grads, device);
    break;
case 11:
    backward_op_unsqueeze(instr, locals, local_grads, device);
    break;
case 15:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_mul<GpuTypes>, 3, 2>, 2, 1, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;
case 18:
    backward_batch_foreach<tensor_foreach<backward_kernel_reduce_product<GpuTypes>, 2, 1, 1>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 99:
    backward_op_matmul(instr, locals, local_grads, device);
    break;
case 100:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_relu<GpuTypes>, 2, 1>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 101:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_leaky_relu<GpuTypes>, 2, 1>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 102:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_elu<GpuTypes>, 2, 1>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 103:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_gelu<GpuTypes>, 2, 1>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 104:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_sigmoid<GpuTypes>, 2, 1>, 1, 1, 0, 1>(instr, locals, local_grads, {}, {0}, device);
    break;
case 105:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_softplus<GpuTypes>, 2, 1>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 106:
    backward_op_rqs_reshape(instr, locals, local_grads, device);
    break;
case 107:
    backward_batch_foreach<tensor_foreach<backward_kernel_rqs_find_bin<GpuTypes>, 5, 4, 2>, 4, 1, 4, 0>(instr, locals, local_grads, {0,1,2,3}, {}, device);
    break;
case 108:
    backward_batch_foreach<tensor_foreach<backward_kernel_rqs_forward<GpuTypes>, 4, 2, 2>, 2, 2, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;
case 109:
    backward_batch_foreach<tensor_foreach<backward_kernel_rqs_inverse<GpuTypes>, 4, 2, 2>, 2, 2, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;
case 110:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_softmax<GpuTypes>, 2, 1>, 1, 1, 0, 1>(instr, locals, local_grads, {}, {0}, device);
    break;
case 111:
    backward_batch_foreach<tensor_foreach<backward_kernel_softmax_prior<GpuTypes>, 2, 2, 1>, 2, 1, 0, 1>(instr, locals, local_grads, {}, {0}, device);
    break;
case 115:
    backward_batch_foreach<tensor_foreach<backward_kernel_sample_discrete_probs_inverse<GpuTypes>, 4, 2, 1>, 2, 2, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;
case 121:
    backward_batch_foreach<tensor_foreach<backward_kernel_select<GpuTypes>, 2, 2, 1>, 2, 1, 1, 0>(instr, locals, local_grads, {1}, {}, device);
    break;
