kernel void init_dp(__global float* zero_row) {
    zero_row[get_global_id(0)] = -INFINITY;
}

kernel void init_N_B(
    __global float* restrict zero_row,
    __global float* restrict first_row,
    float move_score,
    uint N,
    uint B)
{
    first_row[0] = -INFINITY;
    zero_row[N] = 0.0;
    zero_row[B] = move_score;
}

kernel void M_states_handler(
    __global float* restrict dp_cur,
    __global const float* restrict dp_prev,
    __constant float* restrict emissions,
    uint B,
    float B_Mk_score)
{
    size_t cur_col = get_global_id(0) + 1;
    dp_cur[cur_col] = emissions[cur_col] + 
        fmax(dp_prev[cur_col - 1], dp_prev[B] + B_Mk_score);
}

kernel void copy_M(
    __global const float* restrict dp_cur,
    __global float* restrict M_buf,
    uint should_use_M0)
{
    size_t id = get_global_id(0);
    M_buf[id] = dp_cur[id + 1 - should_use_M0];
}

kernel void reduction_step(
    __global float* restrict M_buf,
    uint left_half_size)
{
    size_t id = get_global_id(0);
    M_buf[id] = fmax(M_buf[id], M_buf[left_half_size + id]);
}

kernel void E_J_C_N_B_handler(
    __global float* restrict dp_cur,
    __global const float* restrict dp_prev,
    __global const float* restrict M_buf,
    uint E,
    uint J,
    uint C,
    uint N,
    uint B,
    float loop_score,
    float move_score,
    float E_J_score,
    float E_C_score)
{
    dp_cur[E] = fmax(M_buf[0], M_buf[1]);
    dp_cur[J] = fmax(dp_prev[J] + loop_score, dp_cur[E] + E_J_score);
    dp_cur[C] = fmax(dp_prev[C] + loop_score, dp_cur[E] + E_C_score);
    dp_cur[N] = dp_prev[N] + loop_score;
    dp_cur[B] = fmax(dp_cur[N] + move_score, dp_cur[J] + move_score);
}
