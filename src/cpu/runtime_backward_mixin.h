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
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_mul<CpuTypes>, backward_kernel_mul<SimdTypes>, 3, 2, DeviceType>, 2, 1, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;
case 18:
    backward_batch_foreach<tensor_foreach<backward_kernel_reduce_product<CpuTypes>, backward_kernel_reduce_product<SimdTypes>, 2, 1, 1, DeviceType>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 86:
    backward_op_matmul(instr, locals, local_grads, device);
    break;
case 87:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_relu<CpuTypes>, backward_kernel_relu<SimdTypes>, 2, 1, DeviceType>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 88:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_leaky_relu<CpuTypes>, backward_kernel_leaky_relu<SimdTypes>, 2, 1, DeviceType>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 89:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_elu<CpuTypes>, backward_kernel_elu<SimdTypes>, 2, 1, DeviceType>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 90:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_gelu<CpuTypes>, backward_kernel_gelu<SimdTypes>, 2, 1, DeviceType>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 91:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_sigmoid<CpuTypes>, backward_kernel_sigmoid<SimdTypes>, 2, 1, DeviceType>, 1, 1, 0, 1>(instr, locals, local_grads, {}, {0}, device);
    break;
case 92:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_softplus<CpuTypes>, backward_kernel_softplus<SimdTypes>, 2, 1, DeviceType>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 93:
    backward_op_rqs_reshape(instr, locals, local_grads, device);
    break;
case 94:
    backward_batch_foreach<tensor_foreach<backward_kernel_rqs_find_bin<CpuTypes>, backward_kernel_rqs_find_bin<SimdTypes>, 5, 4, 2, DeviceType>, 4, 1, 4, 0>(instr, locals, local_grads, {0,1,2,3}, {}, device);
    break;
case 95:
    backward_batch_foreach<tensor_foreach<backward_kernel_rqs_forward<CpuTypes>, backward_kernel_rqs_forward<SimdTypes>, 4, 2, 2, DeviceType>, 2, 2, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;
case 96:
    backward_batch_foreach<tensor_foreach<backward_kernel_rqs_inverse<CpuTypes>, backward_kernel_rqs_inverse<SimdTypes>, 4, 2, 2, DeviceType>, 2, 2, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;
case 97:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_softmax<CpuTypes>, backward_kernel_softmax<SimdTypes>, 2, 1, DeviceType>, 1, 1, 0, 1>(instr, locals, local_grads, {}, {0}, device);
    break;
case 98:
    backward_batch_foreach<tensor_foreach<backward_kernel_softmax_prior<CpuTypes>, backward_kernel_softmax_prior<SimdTypes>, 2, 2, 1, DeviceType>, 2, 1, 0, 1>(instr, locals, local_grads, {}, {0}, device);
    break;
case 102:
    backward_batch_foreach<tensor_foreach<backward_kernel_sample_discrete_probs_inverse<CpuTypes>, backward_kernel_sample_discrete_probs_inverse<SimdTypes>, 4, 2, 1, DeviceType>, 2, 2, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;
case 108:
    backward_batch_foreach<tensor_foreach<backward_kernel_select<CpuTypes>, backward_kernel_select<SimdTypes>, 2, 2, 1, DeviceType>, 2, 1, 1, 0>(instr, locals, local_grads, {1}, {}, device);
    break;
