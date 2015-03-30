__kernel void surrogate1d( const float x_in, __global const float *x_recurrent,
			   const float tau, const float pstc_alpha, __global float *pstc_state,
			   __global const float *pc_samples, // pc_n * 1024 + sample_idx
			   __global const float *decoders, // gid(0) * 7 + decoder_idx
			   __global float *decoded_output) {

  int i = get_global_id(0);
  decoded_output[i] = x_in;
  
  float input_prefilter = x_in * tau + x_recurrent[i];
  float input_postfilter = input_prefilter * pstc_alpha + pstc_state[i] * (1.0f - pstc_alpha);
  pstc_state[i] = input_postfilter;
  float pc_output0, pc_output1, pc_output2, pc_output3, pc_output4, pc_output5, pc_output6;
  // assume 1024 PC samples evenly spaced between -2.0 and 2.0
  int samplePoint = (int)floor( (input_postfilter + 2.0f) * (1024.0f / 4.0f) );
  if (samplePoint > 1023) samplePoint = 1023;
  if (samplePoint < 0) samplePoint = 0;
  pc_output0 = pc_samples[0 * 1024 + samplePoint];
  pc_output1 = pc_samples[1 * 1024 + samplePoint];
  pc_output2 = pc_samples[2 * 1024 + samplePoint];
  pc_output3 = pc_samples[3 * 1024 + samplePoint];
  pc_output4 = pc_samples[4 * 1024 + samplePoint];
  pc_output5 = pc_samples[5 * 1024 + samplePoint];
  pc_output6 = pc_samples[6 * 1024 + samplePoint];

  float output = 0.0f;
  output += decoders[i*7 + 0] * pc_output0;
  output += decoders[i*7 + 1] * pc_output1;
  output += decoders[i*7 + 2] * pc_output2;
  output += decoders[i*7 + 3] * pc_output3;
  output += decoders[i*7 + 4] * pc_output4;
  output += decoders[i*7 + 5] * pc_output5;
  output += decoders[i*7 + 6] * pc_output6;

  decoded_output[i] = output;
  
}
