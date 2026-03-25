#include <iostream>
#include <ctime>
#include "include\Core.h"
#include "include\fake_stack.h"
#include "src\tensor_io.tpp"

#define BATCH_SIZE 1

byte ARENA_DATA_1[20000000];
Arena ARENA_1(ARENA_DATA_1, 20000000);
byte ARENA_DATA_2[20000000];
Arena ARENA_2(ARENA_DATA_2, 20000000);


template<int C_IN, int H_IN, int W_IN>
void conv_block_i(TensorMem<_Float16> &input, TensorMem<_Float16> &output, 
                    TensorMem<_Float16> &weight, TensorMem<_Float16> &bias, 
                    TensorMem<_Float16> &gamma, TensorMem<_Float16> &beta, _Float16 epsilon) {
    Conv_Attributes conv_att(1, 1, 1, 3, 3, 1, 0, 0, 1, 2, 2);

    Conv_CNorm_RPad_reflect<_Float16, 3, C_IN, 0>(conv_att, input, weight, bias, output);
    Channel_Norm<_Float16>(output, output, gamma, beta, epsilon, {0, 0, 0, 0}, output.shape);
    Relu(output, output);
}

void conv_block_init(TensorMem<_Float16> &input, TensorMem<_Float16> &output, 
                        TensorMem<_Float16> &gamma_0, TensorMem<_Float16> &beta_0, _Float16 epsilon_0, 
                        TensorMem<_Float16> &weight, TensorMem<_Float16> &bias, 
                        TensorMem<_Float16> &gamma_3, TensorMem<_Float16> &beta_3, _Float16 epsilon_3) {
    // conv_block_init_0
    _Float16* Add_1_output_0_data = ARENA_1.alloc<_Float16>(BATCH_SIZE* 16* 16* 220);
    TensorMem<_Float16> Add_1_output_0(Add_1_output_0_data, input.shape, false);

    Channel_Norm<_Float16>(input, Add_1_output_0, gamma_0, beta_0, epsilon_0, {0, 0, 0, 0}, input.shape);

    Conv_Attributes conv_att(1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1);

    Shape Conv_output_0_shape(BATCH_SIZE, 16, 16, 960);
    _Float16* Conv_output_0_data = ARENA_1.alloc<_Float16>(BATCH_SIZE* 16* 16* 960);
    TensorMem<_Float16> Conv_output_0(Conv_output_0_data, Conv_output_0_shape, false);

    Conv_CNorm_RPad_reflect<_Float16, 3, 220, 0>(conv_att, Add_1_output_0, weight, bias, Conv_output_0);

    // conv_block_init_3
    Channel_Norm<_Float16>(Conv_output_0, output, gamma_3, beta_3, epsilon_3, {0, 0, 0, 0}, output.shape);
    ARENA_1.pop();  // free(Conv_output_0_data)
    ARENA_1.pop();  // free(Add_1_output_0_data)
}

template <int C_IN, int X_SIZE>
void upconv_block_i(TensorMem<_Float16> &input, TensorMem<_Float16> &output, 
                    TensorMem<_Float16> &weight, TensorMem<_Float16> &bias, 
                    TensorMem<_Float16> &gamma, TensorMem<_Float16> &beta, _Float16 epsilon) {
    ConvTranspose_Attributes convt_att(1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 2, 2);

    Shape shape(BATCH_SIZE, X_SIZE* 2, X_SIZE* 2, C_IN / 2);
    _Float16* cvt_data = ARENA_1.alloc<_Float16>(BATCH_SIZE* X_SIZE* X_SIZE* C_IN* 2);
    TensorMem<_Float16> cvt(cvt_data, shape, false);

    ConvTranspose(convt_att, &input, &weight, &bias, &cvt);
    Channel_Norm<_Float16>(cvt, output, gamma, beta, epsilon, {0, 0, 0, 0}, shape);
    ARENA_1.pop();  // free(cvt_data)
    Relu(output, output);
}


void Hific(TensorMem<_Float16> &input, TensorMem<_Float16> &output) {
    _Float16 e = (_Float16) 0.001;

    Shape enc_output_shape(BATCH_SIZE, 16, 16, 220);
    _Float16* enc_output_data = ARENA_2.alloc<_Float16>(BATCH_SIZE* 16* 16* 220);
    TensorMem<_Float16> enc_output(enc_output_data, enc_output_shape, false);
{
    Encoder_Conv_block_1:;

    Shape weight_1_shape(60, 7, 7, 3);
    _Float16* weight_1_data = ARENA_1.alloc<_Float16>(60* 7* 7* 3);
    TensorMem<_Float16> weight_1(weight_1_data, weight_1_shape, false);

    Shape bias_1_shape(1, 1, 1, 60);
    _Float16* bias_1_data = ARENA_1.alloc<_Float16>(60);
    TensorMem<_Float16> bias_1(bias_1_data, bias_1_shape, false);

    Shape conv_1_output_shape(BATCH_SIZE, 256, 256, 60);
    _Float16* conv_1_output_data = ARENA_2.alloc<_Float16>(BATCH_SIZE* 256* 256* 60);
    TensorMem<_Float16> conv_1_output(conv_1_output_data, conv_1_output_shape, false);

    _Float16* gamma_1_data = ARENA_2.alloc<_Float16>(60);
    TensorMem<_Float16> gamma_1(gamma_1_data, bias_1_shape, false);

    _Float16* beta_1_data = ARENA_2.alloc<_Float16>(60);
    TensorMem<_Float16> beta_1(beta_1_data, bias_1_shape, false);

    read_tensor("model_params_2\\Enc_cb1_weight.txt", weight_1);
    read_tensor("model_params_2\\Enc_cb1_bias.txt", bias_1);
    read_tensor("model_params_2\\Enc_cb1_gamma.txt", gamma_1);
    read_tensor("model_params_2\\Enc_cb1_beta.txt", beta_1);

    Conv_Attributes conv_1_att(1, 1, 1, 7, 7, 3, 3, 3, 3, 1, 1);
    Conv_CNorm_RPad_reflect<_Float16, 7, 3, 0>(conv_1_att, input, weight_1, bias_1, conv_1_output);
    ARENA_1.pop();  // bias_1_data
    ARENA_1.pop();  // weight_1_data
    //ARENA_1.pop();  // Pad_output_0_data

    _Float16* norm_1_output_data = ARENA_1.alloc<_Float16>(BATCH_SIZE* 256* 256* 60);
    TensorMem<_Float16> norm_1_output(norm_1_output_data, conv_1_output_shape, false);

    Channel_Norm(conv_1_output, norm_1_output, gamma_1, beta_1, e, {0, 0, 0, 0}, conv_1_output_shape);
    ARENA_2.pop();  // beta_1_data
    ARENA_2.pop();  // gamma_1_data
    ARENA_2.pop();  // conv_1_output_data

    Relu(norm_1_output, norm_1_output);
    



    Encoder_Conv_block_2:;

    Shape weight_2_shape(120, 3, 3, 60);
    _Float16* weight_2_data = ARENA_1.alloc<_Float16>(120* 3* 3* 60);
    TensorMem<_Float16> weight_2(weight_2_data, weight_2_shape, false);

    Shape bias_2_shape(1, 1, 1, 120);
    _Float16* bias_2_data = ARENA_1.alloc<_Float16>(120);
    TensorMem<_Float16> bias_2(bias_2_data, bias_2_shape, false);

    Shape norm_2_output_shape(BATCH_SIZE, 128, 128, 120);
    _Float16* norm_2_output_data = ARENA_2.alloc<_Float16>(BATCH_SIZE* 128* 128* 120);
    TensorMem<_Float16> norm_2_output(norm_2_output_data, norm_2_output_shape, false);

    _Float16* gamma_2_data = ARENA_2.alloc<_Float16>(120);
    TensorMem<_Float16> gamma_2(gamma_2_data, bias_2_shape, false);

    _Float16* beta_2_data = ARENA_2.alloc<_Float16>(120);
    TensorMem<_Float16> beta_2(beta_2_data, bias_2_shape, false);

    read_tensor("model_params_2\\Enc_cb2_weight.txt", weight_2);
    read_tensor("model_params_2\\Enc_cb2_bias.txt", bias_2);
    read_tensor("model_params_2\\Enc_cb2_gamma.txt", gamma_2);
    read_tensor("model_params_2\\Enc_cb2_beta.txt", beta_2);

    conv_block_i<60, 256, 256>(norm_1_output, norm_2_output, weight_2, bias_2, gamma_2, beta_2, e);
    ARENA_1.pop();  // bias_2_data
    ARENA_1.pop();  // weight_2_data
    ARENA_1.pop();  // norm_1_output_data
    ARENA_2.pop();  // beta_2_data
    ARENA_2.pop();  // gamma_2_data
    



    Encoder_Conv_block_3:;

    Shape weight_3_shape(240, 3, 3, 120);
    _Float16* weight_3_data = ARENA_2.alloc<_Float16>(240* 3* 3* 120);
    TensorMem<_Float16> weight_3(weight_3_data, weight_3_shape, false);

    Shape bias_3_shape(1, 1, 1, 240);
    _Float16* bias_3_data = ARENA_2.alloc<_Float16>(240);
    TensorMem<_Float16> bias_3(bias_3_data, bias_3_shape, false);

    Shape norm_3_output_shape(BATCH_SIZE, 64, 64, 240);
    _Float16* norm_3_output_data = ARENA_1.alloc<_Float16>(BATCH_SIZE* 64* 64* 240);
    TensorMem<_Float16> norm_3_output(norm_3_output_data, norm_3_output_shape, false);

    _Float16* gamma_3_data = ARENA_1.alloc<_Float16>(240);
    TensorMem<_Float16> gamma_3(gamma_3_data, bias_3_shape, false);

    _Float16* beta_3_data = ARENA_1.alloc<_Float16>(240);
    TensorMem<_Float16> beta_3(beta_3_data, bias_3_shape, false);

    read_tensor("model_params_2\\Enc_cb3_weight.txt", weight_3);
    read_tensor("model_params_2\\Enc_cb3_bias.txt", bias_3);
    read_tensor("model_params_2\\Enc_cb3_gamma.txt", gamma_3);
    read_tensor("model_params_2\\Enc_cb3_beta.txt", beta_3);

    conv_block_i<120, 128, 128>(norm_2_output, norm_3_output, weight_3, bias_3, gamma_3, beta_3, e);
    ARENA_1.pop();  // beta_3_data
    ARENA_1.pop();  // gamma_3_data
    ARENA_2.pop();  // bias_3_data
    ARENA_2.pop();  // weight_3_data
    ARENA_2.pop();  // norm_2_output_data




    Encoder_Conv_block_4:;

    Shape weight_4_shape(480, 3, 3, 240);
    _Float16* weight_4_data = ARENA_1.alloc<_Float16>(480* 3* 3* 240);
    TensorMem<_Float16> weight_4(weight_4_data, weight_4_shape, false);

    Shape bias_4_shape(1, 1, 1, 480);
    _Float16* bias_4_data = ARENA_1.alloc<_Float16>(480);
    TensorMem<_Float16> bias_4(bias_4_data, bias_4_shape, false);

    Shape norm_4_output_shape(BATCH_SIZE, 32, 32, 480);
    _Float16* norm_4_output_data = ARENA_2.alloc<_Float16>(BATCH_SIZE* 32* 32* 480);
    TensorMem<_Float16> norm_4_output(norm_4_output_data, norm_4_output_shape, false);

    _Float16* gamma_4_data = ARENA_2.alloc<_Float16>(480);
    TensorMem<_Float16> gamma_4(gamma_4_data, bias_4_shape, false);

    _Float16* beta_4_data = ARENA_2.alloc<_Float16>(480);
    TensorMem<_Float16> beta_4(beta_4_data, bias_4_shape, false);

    read_tensor("model_params_2\\Enc_cb4_weight.txt", weight_4);
    read_tensor("model_params_2\\Enc_cb4_bias.txt", bias_4);
    read_tensor("model_params_2\\Enc_cb4_gamma.txt", gamma_4);
    read_tensor("model_params_2\\Enc_cb4_beta.txt", beta_4);

    conv_block_i<240, 64, 64>(norm_3_output, norm_4_output, weight_4, bias_4, gamma_4, beta_4, e);
    ARENA_1.pop();  // bias_4_data
    ARENA_1.pop();  // weight_4_data
    ARENA_1.pop();  // norm_3_output_data
    ARENA_2.pop();  // beta_4_data
    ARENA_2.pop();  // gamma_4_data




    Encoder_Conv_block_5:;

    Shape weight_5_shape(960, 3, 3, 480);
    _Float16* weight_5_data = ARENA_2.alloc<_Float16>(960* 3* 3* 480);
    TensorMem<_Float16> weight_5(weight_5_data, weight_5_shape, false);

    Shape bias_5_shape(1, 1, 1, 960);
    _Float16* bias_5_data = ARENA_2.alloc<_Float16>(960);
    TensorMem<_Float16> bias_5(bias_5_data, bias_5_shape, false);

    Shape norm_5_output_shape(BATCH_SIZE, 16, 16, 960);
    _Float16* norm_5_output_data = ARENA_1.alloc<_Float16>(BATCH_SIZE* 16* 16* 960);
    TensorMem<_Float16> norm_5_output(norm_5_output_data, norm_5_output_shape, false);

    _Float16* gamma_5_data = ARENA_1.alloc<_Float16>(960);
    TensorMem<_Float16> gamma_5(gamma_5_data, bias_5_shape, false);

    _Float16* beta_5_data = ARENA_1.alloc<_Float16>(960);
    TensorMem<_Float16> beta_5(beta_5_data, bias_5_shape, false);

    read_tensor("model_params_2\\Enc_cb5_weight.txt", weight_5);
    read_tensor("model_params_2\\Enc_cb5_bias.txt", bias_5);
    read_tensor("model_params_2\\Enc_cb5_gamma.txt", gamma_5);
    read_tensor("model_params_2\\Enc_cb5_beta.txt", beta_5);

    conv_block_i<480, 32, 32>(norm_4_output, norm_5_output, weight_5, bias_5, gamma_5, beta_5, e);
    ARENA_1.pop();  // beta_5_data
    ARENA_1.pop();  // gamma_5_data
    ARENA_2.pop();  // bias_5_data
    ARENA_2.pop();  // weight_5_data
    ARENA_2.pop();  // norm_4_output_data




    Encoder_Conv_block_out:;

    Shape weight_o_shape(220, 3, 3, 960);
    _Float16* weight_o_data = ARENA_1.alloc<_Float16>(220* 3* 3* 960);
    TensorMem<_Float16> weight_o(weight_o_data, weight_o_shape, false);

    Shape bias_o_shape(1, 1, 1, 220);
    _Float16* bias_o_data = ARENA_1.alloc<_Float16>(220);
    TensorMem<_Float16> bias_o(bias_o_data, bias_o_shape, false);

    read_tensor("model_params_2\\Enc_cbo_weight.txt", weight_o);
    read_tensor("model_params_2\\Enc_cbo_bias.txt", bias_o);

    Conv_Attributes conv_2_att(1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1);
    Conv_CNorm_RPad_reflect<_Float16, 3, 16, 0>(conv_2_att, norm_5_output, weight_o, bias_o, enc_output);
    ARENA_1.pop();  // bias_o_data
    ARENA_1.pop();  // weight_o_data
    ARENA_1.pop();  // norm_5_output_data
}


{
    Hyperprior_Conv_block_1:;

    Shape weight_1_shape(320, 3, 3, 220);
    _Float16* weight_1_data = ARENA_1.alloc<_Float16>(320* 3* 3* 220);
    TensorMem<_Float16> weight_1(weight_1_data, weight_1_shape, false);

    Shape bias_1_shape(1, 1, 1, 320);
    _Float16* bias_1_data = ARENA_1.alloc<_Float16>(320);
    TensorMem<_Float16> bias_1(bias_1_data, bias_1_shape, false);

    Shape conv_1_output_shape(BATCH_SIZE, 16, 16, 320);
    _Float16* conv_1_output_data = ARENA_2.alloc<_Float16>(BATCH_SIZE* 16* 16* 320);
    TensorMem<_Float16> conv_1_output(conv_1_output_data, conv_1_output_shape, false);

    read_tensor("model_params_2\\Hyp_an_c1_weight.txt", weight_1);
    read_tensor("model_params_2\\Hyp_an_c1_bias.txt", bias_1);

    Conv_Attributes conv_1_att(1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1);
    Conv(conv_1_att, &enc_output, &weight_1, &bias_1, &conv_1_output);
    ARENA_1.pop();  // bias_1_data
    ARENA_1.pop();  // weight_1_data

    Relu(conv_1_output, conv_1_output);




    Hyperprior_Conv_block_2:;

    Shape weight_2_shape(320, 5, 5, 320);
    _Float16* weight_2_data = ARENA_2.alloc<_Float16>(320* 5* 5* 320);
    TensorMem<_Float16> weight_2(weight_2_data, weight_2_shape, false);

    _Float16* bias_2_data = ARENA_2.alloc<_Float16>(320);
    TensorMem<_Float16> bias_2(bias_2_data, bias_1_shape, false);

    Shape conv_2_output_shape(BATCH_SIZE, 8, 8, 320);
    _Float16* conv_2_output_data = ARENA_1.alloc<_Float16>(BATCH_SIZE* 8* 8* 320);
    TensorMem<_Float16> conv_2_output(conv_2_output_data, conv_2_output_shape, false);

    read_tensor("model_params_2\\Hyp_an_c2_weight.txt", weight_2);
    read_tensor("model_params_2\\Hyp_an_c2_bias.txt", bias_2);

    Conv_Attributes conv_2_att(1, 1, 1, 5, 5, 2, 2, 2, 2, 2, 2);
    Conv_CNorm_RPad_reflect<_Float16, 5, 16, 0>(conv_2_att, conv_1_output, weight_2, bias_2, conv_2_output);
    ARENA_2.pop();  // bias_2_data
    ARENA_2.pop();  // weight_2_data
    //ARENA_1.pop();  // Pad_1_output_data
    ARENA_2.pop();  // conv_1_output_data

    Relu(conv_2_output, conv_2_output);


    

    Hyperprior_Conv_block_3:;

    _Float16* weight_3_data = ARENA_1.alloc<_Float16>(320* 5* 5* 320);
    TensorMem<_Float16> weight_3(weight_3_data, weight_2_shape, false);

    _Float16* bias_3_data = ARENA_1.alloc<_Float16>(320);
    TensorMem<_Float16> bias_3(bias_3_data, bias_1_shape, false);

    Shape conv_3_output_shape(BATCH_SIZE, 4, 4, 320);
    _Float16* conv_3_output_data = ARENA_2.alloc<_Float16>(BATCH_SIZE* 4* 4* 320);
    TensorMem<_Float16> conv_3_output(conv_3_output_data, conv_3_output_shape, false);

    read_tensor("model_params_2\\Hyp_an_c3_weight.txt", weight_3);
    read_tensor("model_params_2\\Hyp_an_c3_bias.txt", bias_3);

    Conv_CNorm_RPad_reflect<_Float16, 5, 16, 0>(conv_2_att, conv_2_output, weight_3, bias_3, conv_3_output);
    ARENA_1.pop();  // bias_3_data
    ARENA_1.pop();  // weight_3_data
    //ARENA_1.pop();  // Pad_2_output_data
    ARENA_1.pop();  // conv_2_output_data

    Round(conv_3_output, conv_3_output);




    Hyperprior_ConvTrans_block_1:;

    _Float16* weight_4_data = ARENA_2.alloc<_Float16>(320* 5* 5* 320);
    TensorMem<_Float16> weight_4(weight_4_data, weight_2_shape, false);

    _Float16* bias_4_data = ARENA_2.alloc<_Float16>(320);
    TensorMem<_Float16> bias_4(bias_4_data, bias_1_shape, false);

    Shape conv_4_output_shape(BATCH_SIZE, 8, 8, 320);
    _Float16* conv_4_output_data = ARENA_1.alloc<_Float16>(BATCH_SIZE* 8* 8* 320);
    TensorMem<_Float16> conv_4_output(conv_4_output_data, conv_4_output_shape, false);

    read_tensor("model_params_2\\Hyp_sm_c1_weight.txt", weight_4);
    read_tensor("model_params_2\\Hyp_sm_c1_bias.txt", bias_4);

    ConvTranspose_Attributes conv_4_att(1, 1, 1, 5, 5, 1, 1, 2, 2, 2, 2, 2, 2);
    ConvTranspose(conv_4_att, &conv_3_output, &weight_4, &bias_4, &conv_4_output);
    ARENA_2.pop();  // bias_4_data
    ARENA_2.pop();  // weight_4_data
    ARENA_2.pop();  // conv_3_output_data

    Relu(conv_4_output, conv_4_output);




    Hyperprior_ConvTrans_block_2:;

    _Float16* weight_5_data = ARENA_1.alloc<_Float16>(320* 5* 5* 320);
    TensorMem<_Float16> weight_5(weight_5_data, weight_2_shape, false);

    _Float16* bias_5_data = ARENA_1.alloc<_Float16>(320);
    TensorMem<_Float16> bias_5(bias_5_data, bias_1_shape, false);

    Shape conv_5_output_shape(BATCH_SIZE, 16, 16, 320);
    _Float16* conv_5_output_data = ARENA_2.alloc<_Float16>(BATCH_SIZE* 16* 16* 320);
    TensorMem<_Float16> conv_5_output(conv_5_output_data, conv_5_output_shape, false);

    read_tensor("model_params_2\\Hyp_sm_c2_weight.txt", weight_5);
    read_tensor("model_params_2\\Hyp_sm_c2_bias.txt", bias_5);

    ConvTranspose(conv_4_att, &conv_4_output, &weight_5, &bias_5, &conv_5_output);
    ARENA_1.pop();  // bias_5_data
    ARENA_1.pop();  // weight_5_data
    ARENA_1.pop();  // conv_4_output_data

    Relu(conv_5_output, conv_5_output);




    Hyperprior_ConvTrans_block_3:;

    Shape weight_6_shape(220, 3, 3, 320);
    _Float16* weight_6_data = ARENA_2.alloc<_Float16>(220* 3* 3* 320);
    TensorMem<_Float16> weight_6(weight_6_data, weight_6_shape, false);

    Shape bias_6_shape(1, 1, 1, 220);
    _Float16* bias_6_data = ARENA_2.alloc<_Float16>(220);
    TensorMem<_Float16> bias_6(bias_6_data, bias_6_shape, false);

    Shape conv_6_output_shape(BATCH_SIZE, 16, 16, 220);
    _Float16* conv_6_output_data = ARENA_1.alloc<_Float16>(BATCH_SIZE* 16* 16* 220);
    TensorMem<_Float16> conv_6_output(conv_6_output_data, conv_6_output_shape, false);

    read_tensor("model_params_2\\Hyp_sm_c3_weight.txt", weight_6);
    read_tensor("model_params_2\\Hyp_sm_c3_bias.txt", bias_6);

    ConvTranspose_Attributes conv_6_att(1, 1, 1, 3, 3, 0, 0, 1, 1, 1, 1, 1, 1);
    ConvTranspose(conv_6_att, &conv_5_output, &weight_6, &bias_6, &conv_6_output);
    ARENA_2.pop();  // bias_6_data
    ARENA_2.pop();  // weight_6_data
    ARENA_2.pop();  // conv_5_output_data

    Sub(enc_output, conv_6_output, enc_output);
    Round(enc_output, enc_output);
    Add(conv_6_output, enc_output, enc_output);
    ARENA_1.pop();  // conv_6_output
}

{
    Shape flip_shape(BATCH_SIZE, 16, 16, 960);
    _Float16* flip_data = ARENA_1.alloc<_Float16>(BATCH_SIZE* 16* 16* 960);
    TensorMem<_Float16> flip(flip_data, flip_shape, false);




    Generator_Conv_block_init:;

    Shape cbi_norm_0_shape(1, 1, 1, 220);
    _Float16* cbi_gamma_0_data = ARENA_2.alloc<_Float16>(220);
    TensorMem<_Float16> cbi_gamma_0(cbi_gamma_0_data, cbi_norm_0_shape, false);
    _Float16* cbi_beta_0_data = ARENA_2.alloc<_Float16>(220);
    TensorMem<_Float16> cbi_beta_0(cbi_beta_0_data, cbi_norm_0_shape, false);

    Shape cbi_norm_3_shape(1, 1, 1, 960);
    _Float16* cbi_gamma_3_data = ARENA_2.alloc<_Float16>(960);
    TensorMem<_Float16> cbi_gamma_3(cbi_gamma_3_data, cbi_norm_3_shape, false);
    _Float16* cbi_beta_3_data = ARENA_2.alloc<_Float16>(960);
    TensorMem<_Float16> cbi_beta_3(cbi_beta_3_data, cbi_norm_3_shape, false);

    Shape cbi_weight_shape(960, 3, 3, 220);
    _Float16* cbi_weight_data = ARENA_1.alloc<_Float16>(960* 3* 3* 220);
    TensorMem<_Float16> cbi_weight(cbi_weight_data, cbi_weight_shape, false);
    _Float16* cbi_bias_data = ARENA_2.alloc<_Float16>(960);
    TensorMem<_Float16> cbi_bias(cbi_bias_data, cbi_norm_3_shape, false);

    read_tensor("model_params\\Gen_cbi_cbi0_gamma.txt", cbi_gamma_0);
    read_tensor("model_params\\Gen_cbi_cbi0_beta.txt", cbi_beta_0);
    read_tensor("model_params\\Gen_cbi_cbi3_gamma.txt", cbi_gamma_3);
    read_tensor("model_params\\Gen_cbi_cbi3_beta.txt", cbi_beta_3);
    read_tensor("model_params\\Gen_cbi_cbi2_weight.txt", cbi_weight);
    read_tensor("model_params\\Gen_cbi_cbi2_bias.txt", cbi_bias);

    conv_block_init(enc_output, flip, cbi_gamma_0, cbi_beta_0, e, 
                        cbi_weight, cbi_bias, cbi_gamma_3, cbi_beta_3, e);
    ARENA_1.pop();  // free(weight_data)
    ARENA_2.pop();  // free(bias_data)
    ARENA_2.pop();  // free(beta_3_data)
    ARENA_2.pop();  // free(gamma_3_data)
    ARENA_2.pop();  // free(beta_0_data)
    ARENA_2.pop();  // free(gamma_0_data)
    ARENA_2.pop();  // free(enc_output)
    


    
    Generator_Res_block_0_8:;

    _Float16* flep_data = ARENA_2.alloc<_Float16>(BATCH_SIZE* 16* 16* 960);
    TensorMem<_Float16> flep(flep_data, flip_shape, false);
    Identity(flip, flep);

    Shape rb_weight_shape(960, 3, 3, 960);
    Shape rb_bias_shape(1, 1, 1, 960);

    _Float16* rb_weight_1_data = ARENA_1.alloc<_Float16>(960* 3* 3* 960);
    TensorMem<_Float16> rb_weight_1(rb_weight_1_data, rb_weight_shape, false);
    _Float16* rb_bias_1_data = ARENA_1.alloc<_Float16>(960);
    TensorMem<_Float16> rb_bias_1(rb_bias_1_data, rb_bias_shape, false);

    _Float16* rb_weight_2_data = ARENA_2.alloc<_Float16>(960* 3* 3* 960);
    TensorMem<_Float16> rb_weight_2(rb_weight_2_data, rb_weight_shape, false);
    _Float16* rb_bias_2_data = ARENA_2.alloc<_Float16>(960);
    TensorMem<_Float16> rb_bias_2(rb_bias_2_data, rb_bias_shape, false);

    _Float16* rb_gamma_1_data = ARENA_1.alloc<_Float16>(960);
    TensorMem<_Float16> rb_gamma_1(rb_gamma_1_data, rb_bias_shape, false);
    _Float16* rb_beta_1_data = ARENA_1.alloc<_Float16>(960);
    TensorMem<_Float16> rb_beta_1(rb_beta_1_data, rb_bias_shape, false);

    _Float16* rb_gamma_2_data = ARENA_2.alloc<_Float16>(960);
    TensorMem<_Float16> rb_gamma_2(rb_gamma_2_data, rb_bias_shape, false);
    _Float16* rb_beta_2_data = ARENA_2.alloc<_Float16>(960);
    TensorMem<_Float16> rb_beta_2(rb_beta_2_data, rb_bias_shape, false);

    _Float16* flap_data = ARENA_2.alloc<_Float16>(BATCH_SIZE* 16* 16* 960);
    TensorMem<_Float16> flap(flap_data, flip_shape, false);

    char rb_gamma_1_file[] = "model_params\\Gen_rb0_gamma_1.txt";
    char rb_beta_1_file[] = "model_params\\Gen_rb0_beta_1.txt";
    char rb_gamma_2_file[] = "model_params\\Gen_rb0_gamma_2.txt";
    char rb_beta_2_file[] = "model_params\\Gen_rb0_beta_2.txt";
    char rb_weight_1_file[] = "model_params\\Gen_rb0_weight_1.txt";
    char rb_bias_1_file[] = "model_params\\Gen_rb0_bias_1.txt";
    char rb_weight_2_file[] = "model_params\\Gen_rb0_weight_2.txt";
    char rb_bias_2_file[] = "model_params\\Gen_rb0_bias_2.txt";
    for (int i = 0; i < 9; i++) {
        rb_gamma_1_file[19] = rb_beta_1_file[19] = rb_gamma_2_file[19] = rb_beta_2_file[19] 
            = rb_weight_1_file[19] = rb_bias_1_file[19] = rb_weight_2_file[19] = rb_bias_2_file[19] = i + 48;
        Identity(flep, flap);
        read_tensor(rb_gamma_1_file, rb_gamma_1);
        read_tensor(rb_beta_1_file, rb_beta_1);
        read_tensor(rb_gamma_2_file, rb_gamma_2);
        read_tensor(rb_beta_2_file, rb_beta_2);
        read_tensor(rb_weight_1_file, rb_weight_1);
        read_tensor(rb_bias_1_file, rb_bias_1);
        read_tensor(rb_weight_2_file, rb_weight_2);
        read_tensor(rb_bias_2_file, rb_bias_2);

        Resblock___Pad_ref_Conv_11133111111_CNorm_Relu___<16, 960, 960, BATCH_SIZE>
                                (flap, rb_weight_1, rb_bias_1, rb_gamma_1, rb_beta_1, 
                                        rb_weight_2, rb_bias_2, rb_gamma_2, rb_beta_2, e, flep);
    }
    Add(flep, flip, flip);
    ARENA_1.pop();  // free(rb_beta_1_data)
    ARENA_1.pop();  // free(rb_gamma_1_data)
    ARENA_1.pop();  // free(rb_bias_1_data)
    ARENA_1.pop();  // free(rb_weight_1_data)
    ARENA_2.pop();  // free(flap_data)
    ARENA_2.pop();  // free(rb_beta_2_data)
    ARENA_2.pop();  // free(rb_gamma_2_data)
    ARENA_2.pop();  // free(rb_bias_2_data)
    ARENA_2.pop();  // free(rb_weight_2_data)
    ARENA_2.pop();  // free(flep_data)
    


    
    Generator_upconv_block_1_4:;

    Shape upconv_block_1_shape(BATCH_SIZE, 32, 32, 480);
    _Float16* upconv_block_1_data = ARENA_2.alloc<_Float16>(BATCH_SIZE* 32* 32* 480);
    TensorMem<_Float16> upconv_block_1(upconv_block_1_data, upconv_block_1_shape, false);

    Shape ub_weight_1_shape(480, 3, 3, 960);
    Shape ub_bias_1_shape(1, 1, 1, 480);
    _Float16* ub_weight_1_data = ARENA_1.alloc<_Float16>(480* 3* 3* 960);
    TensorMem<_Float16> ub_weight_1(ub_weight_1_data, ub_weight_1_shape, false);
    _Float16* ub_bias_1_data = ARENA_2.alloc<_Float16>(480);
    TensorMem<_Float16> ub_bias_1(ub_bias_1_data, ub_bias_1_shape, false);

    _Float16* ub_gamma_1_data = ARENA_2.alloc<_Float16>(480);
    TensorMem<_Float16> ub_gamma_1(ub_gamma_1_data, ub_bias_1_shape, false);
    _Float16* ub_beta_1_data = ARENA_2.alloc<_Float16>(480);
    TensorMem<_Float16> ub_beta_1(ub_beta_1_data, ub_bias_1_shape, false);

    read_tensor("model_params\\Gen_ucb1_gamma.txt", ub_gamma_1);
    read_tensor("model_params\\Gen_ucb1_beta.txt", ub_beta_1);
    read_tensor("model_params\\Gen_ucb1_weight.txt", ub_weight_1);
    read_tensor("model_params\\Gen_ucb1_bias.txt", ub_bias_1);

    upconv_block_i<960, 16>(flip, upconv_block_1, ub_weight_1, ub_bias_1, ub_gamma_1, ub_beta_1, e);
    ARENA_1.pop();  // free(weight_1_data)
    ARENA_1.pop();  // free(flip_data)
    ARENA_2.pop();  // free(beta_1_data)
    ARENA_2.pop();  // free(gamma_1_data)
    ARENA_2.pop();  // free(bias_1_data)


    Shape upconv_block_2_shape(BATCH_SIZE, 64, 64, 240);
    _Float16* upconv_block_2_data = ARENA_1.alloc<_Float16>(BATCH_SIZE* 64* 64* 240);
    TensorMem<_Float16> upconv_block_2(upconv_block_2_data, upconv_block_2_shape, false);

    Shape ub_weight_2_shape(240, 3, 3, 480);
    Shape ub_bias_2_shape(1, 1, 1, 240);
    _Float16* ub_weight_2_data = ARENA_2.alloc<_Float16>(240* 3* 3* 480);
    TensorMem<_Float16> ub_weight_2(ub_weight_2_data, ub_weight_2_shape, false);
    _Float16* ub_bias_2_data = ARENA_1.alloc<_Float16>(240);
    TensorMem<_Float16> ub_bias_2(ub_bias_2_data, ub_bias_2_shape, false);

    _Float16* ub_gamma_2_data = ARENA_1.alloc<_Float16>(240);
    TensorMem<_Float16> ub_gamma_2(ub_gamma_2_data, ub_bias_2_shape, false);
    _Float16* ub_beta_2_data = ARENA_1.alloc<_Float16>(240);
    TensorMem<_Float16> ub_beta_2(ub_beta_2_data, ub_bias_2_shape, false);

    read_tensor("model_params\\Gen_ucb2_gamma.txt", ub_gamma_2);
    read_tensor("model_params\\Gen_ucb2_beta.txt", ub_beta_2);
    read_tensor("model_params\\Gen_ucb2_weight.txt", ub_weight_2);
    read_tensor("model_params\\Gen_ucb2_bias.txt", ub_bias_2);

    upconv_block_i<480, 32>(upconv_block_1, upconv_block_2, ub_weight_2, ub_bias_2, ub_gamma_2, ub_beta_2, e);
    ARENA_1.pop();  // free(beta_2_data)
    ARENA_1.pop();  // free(gamma_2_data)
    ARENA_1.pop();  // free(bias_2_data)
    ARENA_2.pop();  // free(weight_2_data)
    ARENA_2.pop();  // free(upconv_block_1)


    Shape upconv_block_3_shape(BATCH_SIZE, 128, 128, 120);
    _Float16* upconv_block_3_data = ARENA_2.alloc<_Float16>(BATCH_SIZE* 128* 128* 120);
    TensorMem<_Float16> upconv_block_3(upconv_block_3_data, upconv_block_3_shape, false);

    Shape ub_weight_3_shape(120, 3, 3, 240);
    Shape ub_bias_3_shape(1, 1, 1, 120);
    _Float16* ub_weight_3_data = ARENA_1.alloc<_Float16>(120* 3* 3* 240);
    TensorMem<_Float16> ub_weight_3(ub_weight_3_data, ub_weight_3_shape, false);
    _Float16* ub_bias_3_data = ARENA_2.alloc<_Float16>(120);
    TensorMem<_Float16> ub_bias_3(ub_bias_3_data, ub_bias_3_shape, false);

    _Float16* ub_gamma_3_data = ARENA_2.alloc<_Float16>(120);
    TensorMem<_Float16> ub_gamma_3(ub_gamma_3_data, ub_bias_3_shape, false);
    _Float16* ub_beta_3_data = ARENA_2.alloc<_Float16>(120);
    TensorMem<_Float16> ub_beta_3(ub_beta_3_data, ub_bias_3_shape, false);

    read_tensor("model_params\\Gen_ucb3_gamma.txt", ub_gamma_3);
    read_tensor("model_params\\Gen_ucb3_beta.txt", ub_beta_3);
    read_tensor("model_params\\Gen_ucb3_weight.txt", ub_weight_3);
    read_tensor("model_params\\Gen_ucb3_bias.txt", ub_bias_3);

    upconv_block_i<240, 64>(upconv_block_2, upconv_block_3, ub_weight_3, ub_bias_3, ub_gamma_3, ub_beta_3, e);
    ARENA_2.pop();  // free(beta_3_data)
    ARENA_2.pop();  // free(gamma_3_data)
    ARENA_2.pop();  // free(bias_3_data)
    ARENA_1.pop();  // free(weight_3_data)
    ARENA_1.pop();  // free(upconv_block_2)


    Shape upconv_block_4_shape(BATCH_SIZE, 256, 256, 60);
    _Float16* upconv_block_4_data = ARENA_1.alloc<_Float16>(BATCH_SIZE* 256* 256* 60);
    TensorMem<_Float16> upconv_block_4(upconv_block_4_data, upconv_block_4_shape, false);    

    Shape ub_weight_4_shape(60, 3, 3, 120);
    Shape ub_bias_4_shape(1, 1, 1, 60);
    _Float16* ub_weight_4_data = ARENA_2.alloc<_Float16>(60* 3* 3* 120);
    TensorMem<_Float16> ub_weight_4(ub_weight_4_data, ub_weight_4_shape, false);
    _Float16* ub_bias_4_data = ARENA_1.alloc<_Float16>(60);
    TensorMem<_Float16> ub_bias_4(ub_bias_4_data, ub_bias_4_shape, false);    

    _Float16* ub_gamma_4_data = ARENA_1.alloc<_Float16>(60);
    TensorMem<_Float16> ub_gamma_4(ub_gamma_4_data, ub_bias_4_shape, false);
    _Float16* ub_beta_4_data = ARENA_1.alloc<_Float16>(60);
    TensorMem<_Float16> ub_beta_4(ub_beta_4_data, ub_bias_4_shape, false);
    
    read_tensor("model_params\\Gen_ucb4_gamma.txt", ub_gamma_4);
    read_tensor("model_params\\Gen_ucb4_beta.txt", ub_beta_4);
    read_tensor("model_params\\Gen_ucb4_weight.txt", ub_weight_4);
    read_tensor("model_params\\Gen_ucb4_bias.txt", ub_bias_4);

    upconv_block_i<120, 128>(upconv_block_3, upconv_block_4, ub_weight_4, ub_bias_4, ub_gamma_4, ub_beta_4, e);
    ARENA_1.pop();  // free(beta_4_data)
    ARENA_1.pop();  // free(gamma_4_data)
    ARENA_1.pop();  // free(bias_4_data)
    ARENA_2.pop();  // free(weight_4_data)
    ARENA_2.pop();  // free(upconv_block_3)
    


    Shape flyp_shape(BATCH_SIZE, 256, 256, 3);
    _Float16* flyp_data = ARENA_1.alloc<_Float16>(BATCH_SIZE* 256* 256* 3);
    TensorMem<_Float16> flyp(flyp_data, flyp_shape, false);


    

    Generator_Conv_block_out:;

    Shape cbo_weight_shape(3, 7, 7, 60);
    Shape cbo_bias_shape(1, 1, 1, 3);
    _Float16* cbo_weight_data = ARENA_1.alloc<_Float16>(3* 7* 7* 60);
    TensorMem<_Float16> cbo_weight(cbo_weight_data, cbo_weight_shape, false);
    _Float16* cbo_bias_data = ARENA_1.alloc<_Float16>(3);
    TensorMem<_Float16> cbo_bias(cbo_bias_data, cbo_bias_shape, false);

    read_tensor("model_params\\Gen_cbo_weight.txt", cbo_weight);
    read_tensor("model_params\\Gen_cbo_bias.txt", cbo_bias);

    Conv_Attributes att(1, 1, 1, 7, 7, 3, 3, 3, 3, 1, 1);
    Conv_CNorm_RPad_reflect<_Float16, 7, 60, 0>(att, upconv_block_4, cbo_weight, cbo_bias, flyp);
    Clip<_Float16>(flyp, flyp, 0, 1);
    Cast(flyp, output);
    ARENA_1.pop();  // free(bias_data)
    ARENA_1.pop();  // free(weight_data)
    ARENA_1.pop();  // free(flyp_data)
    ARENA_1.pop();  // free(upconv_block_4)
}
}


int main() {
    clock_t start, end;
    start = clock();

    Shape input_shape(BATCH_SIZE, 256, 256, 3);
    _Float16* input_data = ARENA_1.alloc<_Float16>(BATCH_SIZE* 256* 256* 3);
    TensorMem<_Float16> input(input_data, input_shape, false);
    read_tensor("io_params_2\\input.txt", input);

    Shape output_shape(BATCH_SIZE, 256, 256, 3);
    _Float16* output_data = ARENA_2.alloc<_Float16>(BATCH_SIZE* 256* 256* 3);
    TensorMem<_Float16> output(output_data, output_shape, false);
    
    Hific(input, output);
    end = clock();

    write_tensor("io_params_2\\test_output.txt", output, 20);

    std::cout << "Arena_1 max size: " << ARENA_1.max_runtime_size << "\n" 
                << "Arena_2 max size: " << ARENA_2.max_runtime_size << "\n"
                << "Total runtime: " << ((double) (end - start)) / CLOCKS_PER_SEC << " s\n";

    return 0;
}
