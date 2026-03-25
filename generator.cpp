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


void conv_block_init(TensorMem<_Float16> &input, TensorMem<_Float16> &output, 
                        TensorMem<_Float16> &gamma_0, TensorMem<_Float16> &beta_0, _Float16 epsilon_0, 
                        TensorMem<_Float16> &weight, TensorMem<_Float16> &bias, 
                        TensorMem<_Float16> &gamma_3, TensorMem<_Float16> &beta_3, _Float16 epsilon_3) {
    // conv_block_init_0
    _Float16* Add_1_output_0_data = ARENA_1.alloc<_Float16>(BATCH_SIZE* 16* 16* 220);
    TensorMem<_Float16> Add_1_output_0(Add_1_output_0_data, input.shape, false);

    // Shape reducemean_shape(BATCH_SIZE, 16, 16, 1);
    // _Float16* rdm_data = ARENA_1.alloc<_Float16>(BATCH_SIZE* 16* 16);
    // TensorMem<_Float16> reducemean(rdm_data, reducemean_shape, false);

    // Norm<_Float16>(input, Add_1_output_0, gamma_0, beta_0, epsilon_0, C_AXIS, reducemean);
    // ARENA_1.pop();  // free(rdm_data)
    Channel_Norm<_Float16>(input, Add_1_output_0, gamma_0, beta_0, epsilon_0, {0, 0, 0, 0}, input.shape);


    // prepad
    // int Cast_output_0[8] = {0, 1, 1, 0, 0, 1, 1, 0};

    // Shape Pad_output_0_shape(BATCH_SIZE, 18, 18, 220);
    // _Float16* Pad_output_0_data = ARENA_2.alloc<_Float16>(BATCH_SIZE* 18* 18* 220);
    // TensorMem<_Float16> Pad_output_0(Pad_output_0_data, Pad_output_0_shape, false);

    // Pad<_Float16>(Add_1_output_0, Pad_output_0, Cast_output_0, "reflect", 0);
    // ARENA_1.pop();  // free(Add_1_output_0_data)


    // conv_block_init_2
    // Conv_Attributes conv_att(1, 1, 1, 3, 3, 0, 0, 0, 0, 1, 1);
    Conv_Attributes conv_att(1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1);

    Shape Conv_output_0_shape(BATCH_SIZE, 16, 16, 960);
    _Float16* Conv_output_0_data = ARENA_1.alloc<_Float16>(BATCH_SIZE* 16* 16* 960);
    TensorMem<_Float16> Conv_output_0(Conv_output_0_data, Conv_output_0_shape, false);

    // Conv(conv_att, &Pad_output_0, &weight, &bias, &Conv_output_0);
    // ARENA_2.pop();  // free(Pad_output_0_data)
    Conv_CNorm_RPad_reflect<_Float16, 3, 220, 0>(conv_att, Add_1_output_0, weight, bias, Conv_output_0);


    // conv_block_init_3
    // Shape reducemean_3_shape(BATCH_SIZE, 16, 16, 1);
    // _Float16* rdm_3_data = ARENA_2.alloc<_Float16>(BATCH_SIZE* 16* 16);
    // TensorMem<_Float16> reducemean_3(rdm_3_data, reducemean_3_shape, false);

    // Norm<_Float16>(Conv_output_0, output, gamma_3, beta_3, epsilon_3, C_AXIS, reducemean_3);
    // ARENA_2.pop();  // free(rdm_3_data)
    Channel_Norm<_Float16>(Conv_output_0, output, gamma_3, beta_3, epsilon_3, {0, 0, 0, 0}, output.shape);
    ARENA_1.pop();  // free(Conv_output_0_data)
    ARENA_1.pop();  // free(Add_1_output_0_data)
}

// void res_block_i(TensorMem<_Float16> &input, TensorMem<_Float16> &output, 
//                     TensorMem<_Float16> &weight_1, TensorMem<_Float16> &bias_1, 
//                     TensorMem<_Float16> &gamma_1, TensorMem<_Float16> &beta_1, _Float16 epsilon_1, 
//                     TensorMem<_Float16> &weight_2, TensorMem<_Float16> &bias_2, 
//                     TensorMem<_Float16> &gamma_2, TensorMem<_Float16> &beta_2, _Float16 epsilon_2) {
//     // pad
//     int Cast_output_0[8] = {0, 1, 1, 0, 0, 1, 1, 0};

//     Shape Pad_output_0_shape(BATCH_SIZE, 18, 18, 960);
//     _Float16* Pad_output_0_data = ARENA_2.alloc<_Float16>(BATCH_SIZE* 18* 18* 960);
//     TensorMem<_Float16> Pad_output_0(Pad_output_0_data, Pad_output_0_shape, false);

//     Pad<_Float16>(input, Pad_output_0, Cast_output_0, "reflect", 0);

//     // conv1
//     Conv_Attributes conv_att(1, 1, 1, 3, 3, 0, 0, 0, 0, 1, 1);
//     Conv(conv_att, &Pad_output_0, &weight_1, &bias_1, &input);  // BATCH_SIZE, 16, 16, 960

//     // norm1
//     Shape reducemean_shape(BATCH_SIZE, 16, 16, 1);
//     _Float16* rdm_data = ARENA_1.alloc<_Float16>(BATCH_SIZE* 16* 16);
//     TensorMem<_Float16> reducemean(rdm_data, reducemean_shape, false);

//     Norm<_Float16>(input, output, gamma_1, beta_1, epsilon_1, C_AXIS, reducemean);

//     // Relu
//     Relu(output, output);

//     // pad1
//     Pad<_Float16>(output, Pad_output_0, Cast_output_0, "reflect", 0);

//     // conv2
//     Conv(conv_att, &Pad_output_0, &weight_2, &bias_2, &input);  // BATCH_SIZE, 16, 16, 960
//     ARENA_2.pop();  // free(Pad_output_0_data)

//     // norm2
//     Norm<_Float16>(input, output, gamma_2, beta_2, epsilon_2, C_AXIS, reducemean);
//     ARENA_1.pop();  // free(rdm_data)
// }

template <int C_IN, int X_SIZE>
void upconv_block_i(TensorMem<_Float16> &input, TensorMem<_Float16> &output, 
                    TensorMem<_Float16> &weight, TensorMem<_Float16> &bias, 
                    TensorMem<_Float16> &gamma, TensorMem<_Float16> &beta, _Float16 epsilon) {
    ConvTranspose_Attributes convt_att(1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 2, 2);

    Shape shape(BATCH_SIZE, X_SIZE* 2, X_SIZE* 2, C_IN / 2);
    _Float16* cvt_data = ARENA_1.alloc<_Float16>(BATCH_SIZE* X_SIZE* X_SIZE* C_IN* 2);
    TensorMem<_Float16> cvt(cvt_data, shape, false);

    ConvTranspose_2(convt_att, &input, &weight, &bias, &cvt);

    // shape.C = 1;

    // _Float16* rdm_data = ARENA_2.alloc<_Float16>(BATCH_SIZE* X_SIZE* X_SIZE* 4);
    // TensorMem<_Float16> reducemean(rdm_data, shape, false);

    // Norm<_Float16>(cvt, output, gamma, beta, epsilon, C_AXIS, reducemean);
    // ARENA_2.pop();  // free(rdm_data)
    Channel_Norm<_Float16>(cvt, output, gamma, beta, epsilon, {0, 0, 0, 0}, shape);
    ARENA_1.pop();  // free(cvt_data)

    Relu(output, output);
}

void Hific_generator(TensorMem<_Float16> &input, TensorMem<_Float16> &output) {
    _Float16 e = (_Float16)0.001;

    Shape flip_shape(BATCH_SIZE, 16, 16, 960);
    _Float16* flip_data = ARENA_1.alloc<_Float16>(BATCH_SIZE* 16* 16* 960);
    TensorMem<_Float16> flip(flip_data, flip_shape, false);

    

    // conv_block_init
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

    conv_block_init(input, output, cbi_gamma_0, cbi_beta_0, e, 
                        cbi_weight, cbi_bias, cbi_gamma_3, cbi_beta_3, e);
    ARENA_1.pop();  // free(weight_data)
    ARENA_2.pop();  // free(bias_data)
    ARENA_2.pop();  // free(beta_3_data)
    ARENA_2.pop();  // free(gamma_3_data)
    ARENA_2.pop();  // free(beta_0_data)
    ARENA_2.pop();  // free(gamma_0_data)
    

    
    // res_block_0-8
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


    // _Float16* flop_data = ARENA_1.alloc<_Float16>(BATCH_SIZE* 16* 16* 960);
    // TensorMem<_Float16> flop(flop_data, flip_shape, false);

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
        // res_block_i(flap, flop, rb_weight_1, rb_bias_1, rb_gamma_1, rb_beta_1, e, 
        //                         rb_weight_2, rb_bias_2, rb_gamma_2, rb_beta_2, e);
        Resblock___Pad_ref_Conv_11133111111_CNorm_Relu___<16, 960, 960, BATCH_SIZE>
                                (flap, rb_weight_1, rb_bias_1, rb_gamma_1, rb_beta_1, 
                                        rb_weight_2, rb_bias_2, rb_gamma_2, rb_beta_2, e, output);
        //Add(flep, flop, flep);
    }

    Add(flep, flip, flip);
    //ARENA_1.pop();  // free(flop_data)
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
    


    // Shape flup_shape(BATCH_SIZE, 262, 262, 60);
    // _Float16* flup_data = ARENA_2.alloc<_Float16>(BATCH_SIZE* 262* 262* 60);
    // TensorMem<_Float16> flup(flup_data, flup_shape, false);


    
    // upconv_block_1-4
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

    

    // int Cast_output_0[8] = {0, 3, 3, 0, 0, 3, 3, 0};
    // Pad<_Float16>(upconv_block_4, flup, Cast_output_0, "reflect", 0);
    // ARENA_1.pop();  // free(upconv_block_4)
    


    Shape flyp_shape(BATCH_SIZE, 256, 256, 3);
    _Float16* flyp_data = ARENA_1.alloc<_Float16>(BATCH_SIZE* 256* 256* 3);
    TensorMem<_Float16> flyp(flyp_data, flyp_shape, false);


    
    // conv_block_out
    Shape cbo_weight_shape(3, 7, 7, 60);
    Shape cbo_bias_shape(1, 1, 1, 3);
    _Float16* cbo_weight_data = ARENA_1.alloc<_Float16>(3* 7* 7* 60);
    TensorMem<_Float16> cbo_weight(cbo_weight_data, cbo_weight_shape, false);
    _Float16* cbo_bias_data = ARENA_1.alloc<_Float16>(3);
    TensorMem<_Float16> cbo_bias(cbo_bias_data, cbo_bias_shape, false);

    read_tensor("model_params\\Gen_cbo_weight.txt", cbo_weight);
    read_tensor("model_params\\Gen_cbo_bias.txt", cbo_bias);

    Conv_Attributes att(1, 1, 1, 7, 7, 3, 3, 3, 3, 1, 1);
    //Conv(att, &flup, &cbo_weight, &cbo_bias, &flyp);
    Conv_CNorm_RPad_reflect<_Float16, 7, 60, 0>(att, upconv_block_4, cbo_weight, cbo_bias, flyp);

    Clip<_Float16>(flyp, flyp, 0, 1);

    Cast(flyp, output);
    ARENA_1.pop();  // free(bias_data)
    ARENA_1.pop();  // free(weight_data)
    ARENA_1.pop();  // free(flyp_data)
    //ARENA_2.pop();  // free(flup_data)
    ARENA_1.pop();  // free(upconv_block_4)
}

int main() {
    clock_t start, end;
    start = clock();

    Shape input_shape(BATCH_SIZE, 16, 16, 220);
    _Float16* input_data = ARENA_1.alloc<_Float16>(BATCH_SIZE* 16* 16* 220);
    TensorMem<_Float16> input(input_data, input_shape, false);
    read_tensor("io_params\\Gen_cbi_cbi0_ReduceMean_1.txt", input);

    Shape output_shape(BATCH_SIZE, 256, 256, 3);
    _Float16* output_data = ARENA_2.alloc<_Float16>(BATCH_SIZE* 256* 256* 3);
    TensorMem<_Float16> output(output_data, output_shape, false);
    //Constant_of_shape<_Float16>(output, 0);
    
    Hific_generator(input, output);
    end = clock();

    write_tensor("io_params\\test_output.txt", output, 20);

    std::cout << "Arena_1 max size: " << ARENA_1.max_runtime_size << "\n" 
                << "Arena_2 max size: " << ARENA_2.max_runtime_size << "\n"
                << "Total runtime: " << ((double) (end - start)) / CLOCKS_PER_SEC << " s\n";

    return 0;
}
