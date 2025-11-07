/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#if 0
#include <cstring>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include <pthread.h>

#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/leaf_node.hpp>
#include <rapidsmpf/streaming/core/lineariser.hpp>
#include <rapidsmpf/streaming/core/node.hpp>

#include "base_streaming_fixture.hpp"

using namespace rapidsmpf;
using namespace rapidsmpf::streaming;
namespace node = rapidsmpf::streaming::node;

class StreamingLineariser : public BaseStreamingFixture {
  public:
    // In the following test, a large stack may be required when compiled in debug mode.
    // Therefore, this setup spawns threads with a 64 MiB stack size.
    // See: <https://github.com/rapidsai/rapidsmpf/issues/621>.
    void SetUp() override {
        // Save current default attrs
        pthread_attr_t old_attr;
        pthread_attr_init(&old_attr);
        pthread_getattr_default_np(&old_attr);

        // Set a 64 MiB default stack size for new threads created after this point.
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        constexpr std::size_t big = 1 << 26;
        int err = pthread_attr_setstacksize(&attr, big);
        ASSERT_EQ(err, 0) << "pthread_attr_setstacksize: " << strerror(err);

        err = pthread_setattr_default_np(&attr);
        ASSERT_EQ(err, 0) << "pthread_setattr_default_np: " << strerror(err);

        pthread_attr_destroy(&attr);

        // Create the executor threads, they inherit the big default
        SetUpWithThreads(8);

        // Stash previous default to restore in TearDown
        saved_default_attr_ = old_attr;
    }

    void TearDown() override {
        // Restore previous default.
        if (saved_) {
            pthread_setattr_default_np(&saved_default_attr_);
            pthread_attr_destroy(&saved_default_attr_);
        }
        BaseStreamingFixture::TearDown();
    }

  private:
    pthread_attr_t saved_default_attr_{};
    bool saved_{true};
};

TEST_F(StreamingLineariser, ManyProducers) {
    constexpr std::size_t num_producers = 100;
    constexpr std::size_t num_messages = 30'000;

    auto ch_out = ctx->create_channel();
    auto lineariser = std::make_shared<Lineariser>(ctx, ch_out, num_producers);
    std::vector<Node> tasks;
    tasks.reserve(num_producers + 2);
    auto make_producer = [end = num_messages, stride = num_producers](
                             std::shared_ptr<Context> ctx,
                             std::shared_ptr<Channel> ch_out,
                             std::size_t start
                         ) -> Node {
        for (auto id = start; id < end; id += stride) {
            co_await ctx->executor()->schedule();
            co_await ch_out->send(
                Message{id, std::make_unique<std::size_t>(id), ContentDescription{}}
            );
        }
        co_await ch_out->drain(ctx->executor());
    };
    auto inputs = lineariser->get_inputs();
    EXPECT_EQ(inputs.size(), num_producers);
    for (std::size_t i = 0; i < num_producers; i++) {
        tasks.push_back(make_producer(ctx, inputs[i], i));
    }
    tasks.push_back(lineariser->drain());
    std::vector<Message> outputs;
    outputs.reserve(num_messages);
    tasks.push_back(node::pull_from_channel(ctx, ch_out, outputs));
    run_streaming_pipeline(std::move(tasks));
    EXPECT_EQ(num_messages, outputs.size());
    for (std::size_t i = 0; i < num_messages; i++) {
        EXPECT_EQ(outputs[i].sequence_number(), i);
        EXPECT_EQ(outputs[i].release<std::size_t>(), i);
    }
}
#endif
