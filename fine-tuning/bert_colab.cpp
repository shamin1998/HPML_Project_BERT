#include <torch/script.h> // One-stop header.
#include <torch/data/datasets/base.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <memory>
#include <time.h>
#define BILLION 1000000000L
/*
class CustomDataset : public torch::data::Dataset<CustomDataset> {
private:
  // Declare 2 vectors of tensors for images and labels
    torch::Tensor input_ids;
    torch::Tensor attention_masks;
    torch::Tensor labels;
public:
  // Constructor
  CustomDataset(vector<torch::Tensor> input){
    input_ids = input[0];
    attention_masks = input[1];
    labels=input[2];
  };

  // Override get() function to return tensor at location index
  torch::data::Example<> get(size_t index) {
    torch::Tensor sample_input_ids = input_ids[index];
    torch::Tensor sample_attention_masks= attention_masks[index];
    torch::Tensor sample_labels= labels[index];
    return {sample_input_ids.clone(), sample_attention_masks.clone(),sample_labels.clones()};
  };

  // Return the length of data
  torch::optional<size_t> size() const {
    return labels.size();
  };
};
*/

int main(int argc, const char* argv[]) {

  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    torch::jit::script::Module model = torch::jit::load(argv[2]);
    model.eval();
    torch::jit::script::Module train_data = torch::jit::load(argv[1]);
    
    std::vector<torch::Tensor> inputs;
     for (const auto& p : train_data.parameters()) {
        inputs.push_back(p);
      }
    //auto custom_dataset = CustomDataset(inputs).map(torch::data::transforms::Stack<>());
    //CustomDataset custom_dataset=CustomDataset(inputs)
    const double learning_rate = 0.001;
    torch::jit::parameter_list params = model.parameters();
    std::vector<at::Tensor> paramsVector;
    std::transform(params.begin(), params.end(), std::back_inserter(paramsVector),
                   [](const torch::jit::IValue& ivalue) { return ivalue.toTensor(); });
    
    torch::optim::SGD optimizer(paramsVector, torch::optim::SGDOptions(learning_rate));
    
    model.train();
    std::vector<torch::jit::IValue> inp_inputs{inputs[0]};
    std::vector<torch::jit::IValue> inp_attention{inputs[1]};
    const size_t num_iterations = 10;
    const int64_t batch_size = 32;
    struct timespec start, end;
    
      double time_diff;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(size_t iter=0;iter<num_iterations;++iter){
        
        c10::detail::ListImpl::list_type input_list{inputs[0].index({torch::indexing::Slice(iter*batch_size,(iter+1)*batch_size)}), inputs[1].index({torch::indexing::Slice(iter*batch_size,(iter+1)*batch_size)})};
        
        //at::Tensor output =model.forward(input_list).toTensor();
        auto output=model.forward(input_list);
        //std::cout<<output.toTuple()->elements()[0].toTensor();
        //std::cout<<inputs[2].index({torch::indexing::Slice(0,32)});
        auto loss = torch::nn::functional::cross_entropy(output.toTuple()->elements()[0].toTensor(), inputs[2].index({torch::indexing::Slice(iter*batch_size,(iter+1)*batch_size)}));
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_diff = ((double)end.tv_sec - (double)start.tv_sec) + ((double)end.tv_nsec - (double)start.tv_nsec)/BILLION;
    std::cout<<time_diff;
    

  }
  catch (const c10::Error& e) {
    std::cerr <<e.msg()<< "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";
}
