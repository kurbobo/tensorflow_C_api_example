// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018 - 2019 Daniil Goncharov <neargye@gmail.com>.
//
// Permission is hereby  granted, free of charge, to any  person obtaining a copy
// of this software and associated  documentation files (the "Software"), to deal
// in the Software  without restriction, including without  limitation the rights
// to  use, copy,  modify, merge,  publish, distribute,  sublicense, and/or  sell
// copies  of  the Software,  and  to  permit persons  to  whom  the Software  is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE  IS PROVIDED "AS  IS", WITHOUT WARRANTY  OF ANY KIND,  EXPRESS OR
// IMPLIED,  INCLUDING BUT  NOT  LIMITED TO  THE  WARRANTIES OF  MERCHANTABILITY,
// FITNESS FOR  A PARTICULAR PURPOSE AND  NONINFRINGEMENT. IN NO EVENT  SHALL THE
// AUTHORS  OR COPYRIGHT  HOLDERS  BE  LIABLE FOR  ANY  CLAIM,  DAMAGES OR  OTHER
// LIABILITY, WHETHER IN AN ACTION OF  CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE  OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "tf_utils.hpp"
#include <scope_guard.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <fstream>

static void DeallocateBuffer(void* data, size_t) {
  std::free(data);
}

static TF_Buffer* ReadBufferFromFile(const char* file) {
  std::ifstream f(file, std::ios::binary);
  SCOPE_EXIT{ f.close(); }; // Auto-delete on scope exit.
  if (f.fail() || !f.is_open()) {
    return nullptr;
  }

  f.seekg(0, std::ios::end);
  auto fsize = f.tellg();
  f.seekg(0, std::ios::beg);

  if (fsize < 1) {
    return nullptr;
  }

  auto data = static_cast<char*>(std::malloc(fsize));
  f.read(data, fsize);

  auto buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = DeallocateBuffer;

  return buf;

}


int main() {
  auto buffer = ReadBufferFromFile("saved_model.pb");
  if (buffer == nullptr) {
    std::cout << "Can't read buffer from file" << std::endl;
    return 1;
  }
  const char saved_model_dir[] = "../models/";
  TF_SessionOptions* opt = TF_NewSessionOptions();
  TF_Buffer* run_options = TF_NewBufferFromString("", 0);
  TF_Buffer* metagraph = TF_NewBuffer();
  TF_Status* status = TF_NewStatus();
  const char* tags[] = {"serve"};
  TF_Graph* graph = TF_NewGraph();
  TF_Session* Session = TF_LoadSessionFromSavedModel(
      opt, run_options, saved_model_dir, tags, 1, graph, metagraph, status);

// short BATCH_SIZE = 1;
short VECT_DIM = 20; 
  const std::vector<std::int64_t> input_dims = {1, VECT_DIM};
  const std::vector<int> input_ids = { //101, 9617, 3207, 2474, 12367, 21736, 3972, 5675, 9488, 1061, 10722, 2015, 3653, 12734, 10230, 102, 0, 0, 0, 0
    101, 4888, 8224, 7341, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    //20 values
  };
  const std::vector<int> input_mask = { //1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0
    1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    //20 values
  };
  const std::vector<int> segment_ids = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    //20 values
  };

  int NumInputs = 3;
  TF_Output* Input =(TF_Output*) malloc(sizeof(TF_Output) * NumInputs);
  TF_Output t0 = {TF_GraphOperationByName(graph, "input_ids"), 0};
  TF_Output t1 = {TF_GraphOperationByName(graph, "input_mask"), 0};
  TF_Output t2 = {TF_GraphOperationByName(graph, "segment_ids"), 0};
  Input[0] = t0;
  Input[1] = t1;
  Input[2] = t2;

  int NumOutputs = 2;
  TF_Output* Output =(TF_Output*) malloc(sizeof(TF_Output) * NumOutputs);
  TF_Output t3 = {TF_GraphOperationByName(graph, "logits_intent"), 0};
  TF_Output t4 = {TF_GraphOperationByName(graph, "logits_slot"), 0};
  Output[0] = t3;
  Output[1] = t4;

  TF_Tensor** InputValues =(TF_Tensor**) malloc(sizeof(TF_Tensor*)*NumInputs);
  TF_Tensor** OutputValues =(TF_Tensor**) malloc(sizeof(TF_Tensor*)*NumOutputs);
  /* create tensors with data here */
  auto tensor0 = tf_utils::CreateTensor(TF_INT32, input_dims, input_ids);
  auto tensor1 = tf_utils::CreateTensor(TF_INT32, input_dims, input_mask);
  auto tensor2 = tf_utils::CreateTensor(TF_INT32, input_dims, segment_ids);

  InputValues[0] = tensor0;
  InputValues[1] = tensor1;
  InputValues[2] = tensor2;

  // auto status = TF_NewStatus();
  SCOPE_EXIT{ TF_DeleteStatus(status); }; // Auto-delete on scope exit.
  // auto options = TF_NewSessionOptions();
  TF_DeleteSessionOptions(opt);

  if (TF_GetCode(status) != TF_OK) {
    return 4;
  }
TF_SessionRun(Session, NULL, 
  Input, InputValues, NumInputs, 
  Output, OutputValues, NumOutputs, 
  NULL, 0, 
  NULL,
  status
  );
auto logits_intent = static_cast<float*>(TF_TensorData(OutputValues[0]));
auto logits_slot = static_cast<float*>(TF_TensorData(OutputValues[1]));
int num_max = 0;
float max = logits_intent[0];
for (int i = 0; i < 69; i++) {
  if(logits_intent[i]>max){
    max = logits_intent[i];
  num_max = i;
}
  std::cout << "logits_intent: "<< logits_intent[i]<< " "<< i+1<< std::endl;
}
std::cout << max << std::endl;
std::cout << num_max;

num_max = 0;
max = logits_slot[0];
for (int i = 0; i < 69; i++) {
  if(logits_slot[i]>max){
    max = logits_slot[i];
  num_max = i;
}
  std::cout << "logits_slot: "<< logits_slot[i]<< " "<< i+1<< std::endl;
}
std::cout << max << std::endl;
std::cout << num_max;
//   TF_Tensor* logits_intent = OutputValues[0]; // assuming we want the first one
  
// float* logits_intent_data = (float*)TF_TensorData(logits_intent);
// for (int i=0;i<69;i++){
//   std::cout << logits_intent_data[i]<< std::endl;
// }
// TF_Tensor* logits_slot = OutputValues[1];
// float* logits_slot_data = (float*)TF_TensorData(logits_slot);
// for (int i=0;i<23;i++){
//   std::cout << logits_slot_data[i]<< std::endl;
// }
  return 0;
}
