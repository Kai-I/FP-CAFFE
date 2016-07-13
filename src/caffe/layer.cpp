#include <boost/thread.hpp>
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void Layer<Dtype>::InitMutex() {
  forward_mutex_.reset(new boost::mutex());
}

template <typename Dtype>
void Layer<Dtype>::Lock() {
  if (IsShared()) {
    forward_mutex_->lock();
  }
}

template <typename Dtype>
void Layer<Dtype>::Unlock() {
  if (IsShared()) {
    forward_mutex_->unlock();
  }
}

template <typename Dtype>
void Layer<Dtype>::FixData(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom.size() > 0) {
    input_fixed_pos = bottom[0]->FixPos(input_fixed_width);
  }
  if (top.size() > 0) {
    output_fixed_pos = top[0]->FixPos(output_fixed_width);
  }
}

INSTANTIATE_CLASS(Layer);

}  // namespace caffe
