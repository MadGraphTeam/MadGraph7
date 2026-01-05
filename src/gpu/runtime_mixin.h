// This file was automatically generated based on instruction_set.yaml
// Do not modify its content directly

case 0:
    op_stack(instr, locals, device);
    break;
case 1:
    op_unstack(instr, locals, device);
    break;
case 2:
    op_unstack_sizes(instr, locals, device);
    break;
case 3:
    op_pop(instr, locals, device);
    break;
case 4:
    op_batch_cat(instr, locals, device);
    break;
case 5:
    op_batch_split(instr, locals, device);
    break;
case 6:
    op_cat(instr, locals, device);
    break;
case 7:
    op_batch_size(instr, locals, device);
    break;
case 8:
    op_offset_indices(instr, locals, device);
    break;
case 9:
    op_full(instr, locals, device);
    break;
case 10:
    op_squeeze(instr, locals, device);
    break;
case 11:
    op_unsqueeze(instr, locals, device);
    break;
case 12:
    batch_foreach<tensor_foreach_dynamic<kernel_add<GpuTypes>, 2, 1>, 2, 1>(instr, locals, device);
    break;
case 13:
    batch_foreach<tensor_foreach_dynamic<kernel_add_int<GpuTypes>, 2, 1>, 2, 1>(instr, locals, device);
    break;
case 14:
    batch_foreach<tensor_foreach_dynamic<kernel_sub<GpuTypes>, 2, 1>, 2, 1>(instr, locals, device);
    break;
case 15:
    batch_foreach<tensor_foreach_dynamic<kernel_mul<GpuTypes>, 2, 1>, 2, 1>(instr, locals, device);
    break;
case 16:
    batch_foreach<tensor_foreach_dynamic<kernel_reduce_sum<GpuTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 17:
    batch_foreach<tensor_foreach_dynamic<kernel_reduce_sum_vector<GpuTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 18:
    batch_foreach<tensor_foreach<kernel_reduce_product<GpuTypes>, 1, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 19:
    batch_foreach<tensor_foreach<kernel_sqrt<GpuTypes>, 1, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 20:
    batch_foreach<tensor_foreach<kernel_square<GpuTypes>, 1, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 21:
    batch_foreach<tensor_foreach_dynamic<kernel_min<GpuTypes>, 2, 1>, 2, 1>(instr, locals, device);
    break;
case 22:
    batch_foreach<tensor_foreach_dynamic<kernel_max<GpuTypes>, 2, 1>, 2, 1>(instr, locals, device);
    break;
case 23:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_e<GpuTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 24:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_px<GpuTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 25:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_py<GpuTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 26:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_pz<GpuTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 27:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_mass<GpuTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 28:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_pt<GpuTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 29:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_p_mag<GpuTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 30:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_phi<GpuTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 31:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_theta<GpuTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 32:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_y<GpuTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 33:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_y_abs<GpuTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 34:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_eta<GpuTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 35:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_eta_abs<GpuTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 36:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_delta_eta<GpuTypes>, 2, 1>, 2, 1>(instr, locals, device);
    break;
case 37:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_delta_phi<GpuTypes>, 2, 1>, 2, 1>(instr, locals, device);
    break;
case 38:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_delta_r<GpuTypes>, 2, 1>, 2, 1>(instr, locals, device);
    break;
case 39:
    batch_foreach<tensor_foreach<kernel_boost_beam<GpuTypes>, 3, 1, 1>, 3, 1>(instr, locals, device);
    break;
case 40:
    batch_foreach<tensor_foreach<kernel_boost_beam_inverse<GpuTypes>, 3, 1, 1>, 3, 1>(instr, locals, device);
    break;
case 41:
    batch_foreach<tensor_foreach<kernel_com_p_in<GpuTypes>, 1, 2, 1>, 1, 2>(instr, locals, device);
    break;
case 42:
    batch_foreach<tensor_foreach<kernel_r_to_x1x2<GpuTypes>, 3, 3, 1>, 3, 3>(instr, locals, device);
    break;
case 43:
    batch_foreach<tensor_foreach<kernel_x1x2_to_r<GpuTypes>, 3, 2, 1>, 3, 2>(instr, locals, device);
    break;
case 44:
    batch_foreach<tensor_foreach<kernel_diff_cross_section<GpuTypes>, 6, 1, 1>, 6, 1>(instr, locals, device);
    break;
case 45:
    batch_foreach<tensor_foreach<kernel_two_body_decay_com<GpuTypes>, 5, 3, 1>, 5, 3>(instr, locals, device);
    break;
case 46:
    batch_foreach<tensor_foreach<kernel_two_body_decay<GpuTypes>, 6, 3, 1>, 6, 3>(instr, locals, device);
    break;
case 47:
    batch_foreach<tensor_foreach<kernel_two_to_two_particle_scattering_com<GpuTypes>, 6, 3, 1>, 6, 3>(instr, locals, device);
    break;
case 48:
    batch_foreach<tensor_foreach<kernel_two_to_two_particle_scattering<GpuTypes>, 6, 3, 1>, 6, 3>(instr, locals, device);
    break;
case 49:
    batch_foreach<tensor_foreach<kernel_two_to_three_particle_scattering<GpuTypes>, 8, 3, 1>, 8, 3>(instr, locals, device);
    break;
case 50:
    batch_foreach<tensor_foreach<kernel_three_body_decay_com<GpuTypes>, 9, 4, 1>, 9, 4>(instr, locals, device);
    break;
case 51:
    batch_foreach<tensor_foreach<kernel_three_body_decay<GpuTypes>, 10, 4, 1>, 10, 4>(instr, locals, device);
    break;
case 52:
    batch_foreach<tensor_foreach<kernel_t_inv_min_max<GpuTypes>, 4, 2, 1>, 4, 2>(instr, locals, device);
    break;
case 53:
    batch_foreach<tensor_foreach<kernel_s_inv_min_max<GpuTypes>, 6, 2, 1>, 6, 2>(instr, locals, device);
    break;
case 54:
    batch_foreach<tensor_foreach<kernel_invariants_from_momenta<GpuTypes>, 2, 1, 1>, 2, 1>(instr, locals, device);
    break;
case 55:
    batch_foreach<tensor_foreach<kernel_sde2_channel_weights<GpuTypes>, 4, 1, 1>, 4, 1>(instr, locals, device);
    break;
case 56:
    batch_foreach<tensor_foreach<kernel_subchannel_weights<GpuTypes>, 6, 1, 1>, 6, 1>(instr, locals, device);
    break;
case 57:
    batch_foreach<tensor_foreach<kernel_apply_subchannel_weights<GpuTypes>, 4, 1, 1>, 4, 1>(instr, locals, device);
    break;
case 58:
    batch_foreach<tensor_foreach<kernel_pt_eta_phi_x<GpuTypes>, 3, 1, 1>, 3, 1>(instr, locals, device);
    break;
case 59:
    batch_foreach<tensor_foreach<kernel_mirror_momenta<GpuTypes>, 2, 1, 1>, 2, 1>(instr, locals, device);
    break;
case 60:
    batch_foreach<tensor_foreach<kernel_momenta_to_x1x2<GpuTypes>, 2, 2, 1>, 2, 2>(instr, locals, device);
    break;
case 61:
    batch_foreach<tensor_foreach<kernel_uniform_invariant<GpuTypes>, 3, 2, 1>, 3, 2>(instr, locals, device);
    break;
case 62:
    batch_foreach<tensor_foreach<kernel_uniform_invariant_inverse<GpuTypes>, 3, 2, 1>, 3, 2>(instr, locals, device);
    break;
case 63:
    batch_foreach<tensor_foreach<kernel_breit_wigner_invariant<GpuTypes>, 5, 2, 1>, 5, 2>(instr, locals, device);
    break;
case 64:
    batch_foreach<tensor_foreach<kernel_breit_wigner_invariant_inverse<GpuTypes>, 5, 2, 1>, 5, 2>(instr, locals, device);
    break;
case 65:
    batch_foreach<tensor_foreach<kernel_stable_invariant<GpuTypes>, 4, 2, 1>, 4, 2>(instr, locals, device);
    break;
case 66:
    batch_foreach<tensor_foreach<kernel_stable_invariant_inverse<GpuTypes>, 4, 2, 1>, 4, 2>(instr, locals, device);
    break;
case 67:
    batch_foreach<tensor_foreach<kernel_stable_invariant_nu<GpuTypes>, 5, 2, 1>, 5, 2>(instr, locals, device);
    break;
case 68:
    batch_foreach<tensor_foreach<kernel_stable_invariant_nu_inverse<GpuTypes>, 5, 2, 1>, 5, 2>(instr, locals, device);
    break;
case 69:
    batch_foreach<tensor_foreach<kernel_fast_rambo_massless<GpuTypes>, 3, 2, 1>, 3, 2>(instr, locals, device);
    break;
case 70:
    batch_foreach<tensor_foreach<kernel_fast_rambo_massless_com<GpuTypes>, 2, 2, 1>, 2, 2>(instr, locals, device);
    break;
case 71:
    batch_foreach<tensor_foreach<kernel_fast_rambo_massive<GpuTypes>, 4, 2, 1>, 4, 2>(instr, locals, device);
    break;
case 72:
    batch_foreach<tensor_foreach<kernel_fast_rambo_massive_com<GpuTypes>, 3, 2, 1>, 3, 2>(instr, locals, device);
    break;
case 73:
    batch_foreach<tensor_foreach<kernel_cut_unphysical<GpuTypes>, 4, 1, 1>, 4, 1>(instr, locals, device);
    break;
case 74:
    batch_foreach<tensor_foreach<kernel_cut_one<GpuTypes>, 3, 1, 1>, 3, 1>(instr, locals, device);
    break;
case 75:
    batch_foreach<tensor_foreach<kernel_cut_all<GpuTypes>, 3, 1, 1>, 3, 1>(instr, locals, device);
    break;
case 76:
    batch_foreach<tensor_foreach<kernel_cut_any<GpuTypes>, 3, 1, 1>, 3, 1>(instr, locals, device);
    break;
case 77:
    batch_foreach<tensor_foreach<kernel_scale_transverse_energy<GpuTypes>, 1, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 78:
    batch_foreach<tensor_foreach<kernel_scale_transverse_mass<GpuTypes>, 1, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 79:
    batch_foreach<tensor_foreach<kernel_scale_half_transverse_mass<GpuTypes>, 1, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 80:
    batch_foreach<tensor_foreach<kernel_scale_partonic_energy<GpuTypes>, 1, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 81:
    batch_foreach<tensor_foreach<kernel_chili_forward<GpuTypes>, 5, 4, 1>, 5, 4>(instr, locals, device);
    break;
case 82:
    op_matrix_element(instr, locals, device);
    break;
case 83:
    batch_foreach<tensor_foreach<kernel_collect_channel_weights<GpuTypes>, 3, 1, 1>, 3, 1>(instr, locals, device);
    break;
case 84:
    batch_foreach<tensor_foreach<kernel_interpolate_pdf<GpuTypes>, 6, 1, 1>, 6, 1>(instr, locals, device);
    break;
case 85:
    batch_foreach<tensor_foreach<kernel_interpolate_alpha_s<GpuTypes>, 3, 1, 1>, 3, 1>(instr, locals, device);
    break;
case 86:
    op_matmul(instr, locals, device);
    break;
case 87:
    batch_foreach<tensor_foreach_dynamic<kernel_relu<GpuTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 88:
    batch_foreach<tensor_foreach_dynamic<kernel_leaky_relu<GpuTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 89:
    batch_foreach<tensor_foreach_dynamic<kernel_elu<GpuTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 90:
    batch_foreach<tensor_foreach_dynamic<kernel_gelu<GpuTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 91:
    batch_foreach<tensor_foreach_dynamic<kernel_sigmoid<GpuTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 92:
    batch_foreach<tensor_foreach_dynamic<kernel_softplus<GpuTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 93:
    op_rqs_reshape(instr, locals, device);
    break;
case 94:
    batch_foreach<tensor_foreach<kernel_rqs_find_bin<GpuTypes>, 4, 1, 2>, 4, 1>(instr, locals, device);
    break;
case 95:
    batch_foreach<tensor_foreach<kernel_rqs_forward<GpuTypes>, 2, 2, 2>, 2, 2>(instr, locals, device);
    break;
case 96:
    batch_foreach<tensor_foreach<kernel_rqs_inverse<GpuTypes>, 2, 2, 2>, 2, 2>(instr, locals, device);
    break;
case 97:
    batch_foreach<tensor_foreach_dynamic<kernel_softmax<GpuTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 98:
    batch_foreach<tensor_foreach<kernel_softmax_prior<GpuTypes>, 2, 1, 1>, 2, 1>(instr, locals, device);
    break;
case 99:
    batch_foreach<tensor_foreach<kernel_sample_discrete<GpuTypes>, 2, 2, 1>, 2, 2>(instr, locals, device);
    break;
case 100:
    batch_foreach<tensor_foreach<kernel_sample_discrete_inverse<GpuTypes>, 2, 2, 1>, 2, 2>(instr, locals, device);
    break;
case 101:
    batch_foreach<tensor_foreach<kernel_sample_discrete_probs<GpuTypes>, 2, 2, 1>, 2, 2>(instr, locals, device);
    break;
case 102:
    batch_foreach<tensor_foreach<kernel_sample_discrete_probs_inverse<GpuTypes>, 2, 2, 1>, 2, 2>(instr, locals, device);
    break;
case 103:
    op_discrete_histogram(instr, locals, device);
    break;
case 104:
    batch_foreach<tensor_foreach<kernel_permute_momenta<GpuTypes>, 3, 1, 1>, 3, 1>(instr, locals, device);
    break;
case 105:
    batch_foreach<tensor_foreach<kernel_gather<GpuTypes>, 2, 1, 1>, 2, 1>(instr, locals, device);
    break;
case 106:
    batch_foreach<tensor_foreach<kernel_gather_int<GpuTypes>, 2, 1, 1>, 2, 1>(instr, locals, device);
    break;
case 107:
    batch_foreach<tensor_foreach<kernel_select_int<GpuTypes>, 2, 1, 1>, 2, 1>(instr, locals, device);
    break;
case 108:
    batch_foreach<tensor_foreach<kernel_select<GpuTypes>, 2, 1, 1>, 2, 1>(instr, locals, device);
    break;
case 109:
    batch_foreach<tensor_foreach<kernel_select_vector<GpuTypes>, 2, 1, 1>, 2, 1>(instr, locals, device);
    break;
case 110:
    batch_foreach<tensor_foreach<kernel_argsort<GpuTypes>, 1, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 111:
    batch_foreach<tensor_foreach<kernel_one_hot<GpuTypes>, 2, 1, 1>, 2, 1>(instr, locals, device);
    break;
case 112:
    op_nonzero(instr, locals, device);
    break;
case 113:
    op_batch_gather(instr, locals, device);
    break;
case 114:
    op_batch_scatter(instr, locals, device);
    break;
case 115:
    op_random(instr, locals, device);
    break;
case 116:
    op_unweight(instr, locals, device);
    break;
case 117:
    batch_foreach<tensor_foreach<kernel_vegas_forward<GpuTypes>, 2, 2, 2>, 2, 2>(instr, locals, device);
    break;
case 118:
    batch_foreach<tensor_foreach<kernel_vegas_inverse<GpuTypes>, 2, 2, 2>, 2, 2>(instr, locals, device);
    break;
case 119:
    op_vegas_histogram(instr, locals, device);
    break;
case 120:
    op_histogram(instr, locals, device);
    break;
