#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/conv_layer.hpp"


using namespace std;
using namespace caffe;

void preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels) {
	/* Convert the input image to the input image format of the network. */

}

template<typename Dtype>
void top_k(const Dtype* data, int n, int k) {
	priority_queue<pair<Dtype, int> > q;
	for (int i = 0; i < n; ++i) {
		q.push(pair<Dtype, int>(data[i], i));
	}
	for (int i = 0; i < k; ++i) {
		pair<Dtype, int> ki = q.top();
		std::cout << "index[" << i << "] = " << ki.second << " @ " << ki.first << std::endl;
		q.pop();
	}
}

template<typename Dtype>
int getLabel(const Dtype* data, int n) {
	priority_queue<pair<Dtype, int> > q;
	for (int i = 0; i < n; ++i) {
		q.push(pair<Dtype, int>(data[i], i));
	}
	int ki = q.top().second;
	q.pop();
	return ki;
}



int main(int argc, char** argv) {

	if (argc != 4) {
		std::cerr << "Usage: " << argv[0]
			<< " deploy.prototxt network.caffemodel"
			<< " round_number" << std::endl;
		return 1;
	}

	string model_file = argv[1];
	string trained_file = argv[2];
	string round_string = argv[3];



	//string model_file = "Vgg16.prototxt";
	//string trained_file = "Vgg16.caffemodel";
	//string model_file = "SqueezeNet.prototxt";
	//string trained_file = "SqueezeNet.caffemodel";
	string img_file = "3.jpg";

	Net<float>* net = new Net<float>(model_file, TEST);
	net->CopyTrainedLayersFromBinaryProto(trained_file);

	int round;
	stringstream ss;
	ss << round_string;
	ss >> round;

	float fixacctotal_5 = 0;
	float floatacctotal_5 = 0;
	float fixacctotal_1 = 0;
	float floatacctotal_1 = 0;

	int layer_size = net->layers().size();

	//run a total float net
	net->FixSetup(0);
	net->Forward();

	int show_layer = 73;

	vector<Blob<float>*> output_label = net->top_vecs()[show_layer];
	const float* output_d_label = output_label[0]->cpu_data();
	
	ofstream file("output_float.txt", ios::out);

	for (int i = 0; i < 10000; i++){
		file << output_d_label[i] << endl;
	}

	file.close();
	//fix the net with 8 bits
	net->FixSetup(8);
	net->Fix();
	//save the parameter
	net->SaveFixInfo("info.txt");
	
	
	NetParameter net_param;
	net->ToProto(&net_param, false);
	string filename = "suibian.caffemodel";
	WriteProtoToBinaryFile(net_param, filename.c_str());


	net->LoadFixInfo("info.txt");
	
	net->SaveFixInfo("check.txt");
	
	return 0;
}

