template <typename batch_t, typename output_mdspan_t>
struct future {
  template <typename mdspan_t>
  auto set_result(raft::device_resources const& res, mdspan_t result) {
    raft::copy(res, output, result);
  }
  auto set_ready() {
    auto lock = std::unique_lock{mtx};
    is_ready.exchange(true);
  }
  auto get(raft::device_resources const& res) {
    while (true){
      auto lock = std::unique_lock{mtx};
      if (is_ready) {
        break;
      }
      auto shared_batch = batch.lock();
      if (shared_batch) {
        shared_batch->execute(res);
      }
      std::this_thread::yield();
    }
    return output;
  }
 private:
  std::shared_ptr<batch_t> batch;
  output_mdspan_t output;
  std::atomic<bool> is_ready = false;
  std::mutex mtx;
};

template<typename input_mdarray_t, typename output_mdarray_t, typename lambda_t>
struct batch {
  template <typename suboutput_mdspan_t, typename subinput_mdspan_t>
  auto put(raft::device_resources const& res, subinput_mdspan_t subinput, suboutput_mdspan_t suboutput) {
    if (subinput.extent(0) > batch_size) {
      // TODO: Check that input is on expected device and then directly call
      // lambda on subinput and suboutput
    } else {
      while (true) {
        while (cur_row.load() + subinput->extent(0) > batch_size) {
          if (!is_executing.exchange(true)) {
            execute(res);
          } else {
            std::this_thread::yield();
          }
        }
        auto lock = std::unique_lock{mtx};
        if (cur_row.load() + subinput->extent(0) <= batch_size) {
          cur_row += subinput->extent(0);
          // TODO: Copy subinput to input
          // TODO: Return future
          break;
        }
      }
    }
  }
  void execute(raft::device_resources const& res) {
    if (!is_executing.exchange(true)) {
      // Prevent any more calls to put until execution is complete
      auto lock = std::unique_lock{mtx};
      if (cur_row.load() > 0) {
        // Call lambda
        // Clear input
        // Set result for all stored futures, then sync resources
      }
    }
  }
  ~batch() {
    if (cur_row.load() > 0) {
      try {
        execute();
      } catch (...) {
        RAFT_WARN("Batch execution failed during destruction");
      }
    }
  }
 private:
  streamsafe_wrapper<input_mdarray_t> input;
  streamsafe_wrapper<output_mdarray_t> output;
  std::size_t batch_size;  // Have to store this so we're not constantly
                           // reading from output object
  lambda_t lambda;
  std::mutex mtx;
  std::atomic<std::size_t> cur_row;
  std::atomic<bool> is_executing;
};
