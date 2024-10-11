#include <iostream>
#include <math.h>
#include <string>
#include <iomanip>



enum class RepetitionPenaltyType {
    Additive,        // the presence penalty
    Multiplicative,  // the repetition penalty
    None             // No repetition penalty.
};

#define FLT_MAX		__FLT_MAX__
#define FLT_MIN		__FLT_MIN__

void cpuInvokeBanBadWords(
    float *logits,
    int *output_ids,
    int *bad_words,
    int bad_words_len,
    bool share_words,         // 不同 batch 是否共享不良单词的标志
    int batch_size,
    int vocab_size_padded,
    int step
){
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        const int* base_bad_words         = share_words ? bad_words : bad_words + batch_idx * 2 * bad_words_len;
        const int* base_bad_words_offsets = base_bad_words + bad_words_len;

        for (int bad_word_id = 0; bad_word_id < bad_words_len; bad_word_id++) {
            if ( base_bad_words_offsets[bad_word_id] < 0){
                break;
            }

            const int item_end = base_bad_words_offsets[bad_word_id];
            const int item_start = (bad_word_id > 0) ? base_bad_words_offsets[bad_word_id - 1] : 0;
            const int item_size = item_end - item_start;

            /* The single-token case unconditionally bans the token */
            bool should_ban = item_size == 1;

            if (item_size > 1 && step > item_size - 1){
                should_ban = true;
                for (int token_idx = item_size-2; token_idx >= 0; token_idx--) {
                    int previous_token = output_ids[(step - (item_size - 1) + token_idx) * batch_size + batch_idx];
                    if ( previous_token != base_bad_words[item_start + token_idx]){
                        should_ban = false;
                        break;
                    }
                }
            }

            if (should_ban) {
                int banned_token = base_bad_words[item_end - 1];
                if (0 < banned_token && banned_token < vocab_size_padded) {
                    logits[batch_idx * vocab_size_padded + banned_token] = static_cast<float>(-INFINITY);
                }
            }
        }
    } 
}

void cpuInvokeTemperaturePenalty(
    float *logits,
    float *bias,
    float *temperatures,
    int temperatures_size,
    int batch_size,
    int vocab_size,
    int vocab_size_padded
){
    for (int batch_idx=0; batch_idx < batch_size; batch_idx++) {
        float temperature = temperatures_size > 1 ? temperatures[batch_idx] : temperatures[0];
        float inv_temperature = 1.0f / (temperature + 1e-6f);

        for (int vocab_idx = 0; vocab_idx < vocab_size; vocab_idx++){
            int index = batch_idx * vocab_size_padded + vocab_idx;
            float logit = static_cast<float>(logits[index]);
            if (bias != nullptr) {
                logit += static_cast<float>(bias[vocab_idx]);
            }
            logits[index] = static_cast<float>(logit * inv_temperature);
        }
    }

}

void cpuInvokeRepetitionPenalty(
    float *logits,
    int *output_ids,
    int *input_lengths,
    int max_input_length,
    int batch_size,
    int vocab_size,
    int vocab_size_padded,
    int step,
    const RepetitionPenaltyType repetition_penalty_type,
    float repetition_penalty // 重复惩罚系数，用于降低已经生成过的词的概率
)
{
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        // 获取当前对应的输入长度
        const int input_length = (input_lengths != nullptr) ? input_lengths[batch_idx] : max_input_length;
        float repet_penalty = static_cast<float>(repetition_penalty);

        int offset = batch_idx * vocab_size_padded;
        for (int i = step - 1; i >= 0; --i){
            // input pad 的部分
            if ( i >= input_length && i < max_input_length) {
                continue;
            }
            int idx = batch_idx + i * batch_size;
            int token_id = output_ids[idx];

            float logit = static_cast<float>(logits[offset + token_id]);
            switch (repetition_penalty_type) {
                case RepetitionPenaltyType::Additive:
                    logits[offset + token_id] = static_cast<float>(logit - repetition_penalty);
                    break;
                case RepetitionPenaltyType::Multiplicative:
                    logits[offset + token_id] =
                            static_cast<float>(logit < 0.0f ? logit * repetition_penalty : logit / repetition_penalty);
                    break;
                default: throw std::domain_error("Invalid repetition penalty type.");
            }
        }
    }
}

void cpuInvokeMinLengthPenalty(
    float *logits,
    int min_length,
    int *end_ids,
    int *sequence_lengths,
    int max_input_length,
    int batch_size,
    int vocab_size_padded
){
    for (int batch_idx=0; batch_idx < batch_size; batch_idx++){
        // We need +1 because sequence_lengths = max_input_length + num_gen_tokens - 1
        if (sequence_lengths[batch_idx] + 1 - max_input_length < min_length) {
            float mask_val = static_cast<float>(-INFINITY);
            logits[batch_idx * vocab_size_padded + end_ids[batch_idx]] = mask_val;
        }
    }
}


void cpuInvokeAddBiasEndMask(
    float *logits,
    float *bias,
    int *end_ids,
    bool *finished,
    int batch_size,
    int vocab_size,
    int vocab_size_padded
){
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {

        bool finish = finished != nullptr ? finished[batch_idx] : false;
        int offset = batch_idx * vocab_size_padded;

        for (int vocab_idx = 0; vocab_idx < vocab_size_padded; vocab_idx++) {
            if (vocab_idx < vocab_size) {
                if (finish) {
                    logits[offset + vocab_idx] = (vocab_idx == end_ids[batch_idx]) ? FLT_MAX : FLT_MIN;
                }
                else {
                    // possible wrong
                    float bias_val = (bias != nullptr) ? bias[batch_idx] : (float)0.0f;
                    logits[offset + vocab_idx] += bias_val;
                }
            }
            // padded part
            else {
                logits[offset + vocab_idx] = FLT_MIN;
            }
        }
    }
}

void cpuInvokeAddBiasSoftMax(
    float *logits,
    float *bias,
    int *end_ids,
    bool *finished,
    int batch_size,
    int vocab_size,
    int vocab_size_padded
){
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        float max_val = FLT_MIN;
        bool finish = finished != nullptr ? finished[batch_idx] : false;
        int offset = batch_idx * vocab_size_padded;

        for (int vocab_idx = 0; vocab_idx < vocab_size_padded; vocab_idx++) {
            if (vocab_idx < vocab_size) {
                if (finish) {
                    logits[offset + vocab_idx] = (vocab_idx == end_ids[batch_idx]) ? FLT_MAX : FLT_MIN;
                }
                else {
                    // possible wrong
                    float bias_val = (bias != nullptr) ? bias[batch_idx] : (float)0.0f;
                    logits[offset + vocab_idx] += bias_val;
                }
            }
            // padded part
            else {
                logits[offset + vocab_idx] = FLT_MIN;
            }

            float logit = static_cast<float>(logits[offset + vocab_idx]);
            if (logit > max_val) {
                max_val = logit;
            }
        }

        float sum = 0.0f;
        for (int vocab_idx = 0; vocab_idx < vocab_size; vocab_idx++) {
            logits[offset + vocab_idx] = expf(static_cast<float>(logits[offset + vocab_idx]) - max_val);
            sum += logits[offset + vocab_idx];
        }
        for (int vocab_idx = 0; vocab_idx < vocab_size; vocab_idx++) {
            logits[offset + vocab_idx] = ((float)logits[offset + vocab_idx] / (sum + 1e-6f));
        }
    }
}

void bubble_sort_topk(float* vals, int* indices, int n, int k) {
    for (int i = 0; i < k; i++) {
        for (int j = i + 1; j < n; j++) {
            if (vals[j] > vals[i]) {
                // Swap values
                float tmp_val = vals[i];
                vals[i] = vals[j];
                vals[j] = tmp_val;

                // Swap corresponding indices
                int tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
            }
        }
    }
}

// Helper function to generate random float between 0 and 1
float rand_float() {
    return rand() / (float)RAND_MAX;
}

void cpuInvokeBatchTopKSampling(
    float *log_probs,
    int *ids,                       // output_ids + step * batch_size
    int *sequence_lengths,
    int *end_ids,
    bool *finished,
    int max_top_k,
    int* top_ks,
    float top_p,
    float* top_ps,
    int batch_size,
    int vocab_size,
    int vocab_size_padded,
    float *cum_log_probs,
    float *output_log_probs,       // output_log_probs + (step - max_input_length) * batch_size
    bool *skip_decode
) {
    for (int batch_id = 0; batch_id < batch_size; batch_id++){
        if (skip_decode != NULL && skip_decode[batch_id]) {
            continue;
        }
        int k = (top_ks != NULL) ? top_ks[batch_id] : max_top_k;
        float prob_threshold = (top_ps != NULL) ? top_ps[batch_id] : top_p;

        if (finished != NULL && finished[batch_id]){
            ids[batch_id] = end_ids[batch_id];
            continue;
        }

        // Step 1: Perform Top-k selection directly on log_probs
        float topk_vals[vocab_size_padded];
        int topk_indices[vocab_size_padded];
        for (int i = 0; i < vocab_size_padded; ++i) {
            topk_vals[i] = log_probs[batch_id * vocab_size_padded + i];
            topk_indices[i] = i;
        }

        // Sort the first k elements
        bubble_sort_topk(topk_vals, topk_indices, vocab_size_padded, k);

        // Step 2: Softmax 前处理，找到最大值
        // if cum_log_probs are computed, this operation is already pre-processed
        float s_max = FLT_MIN;
        if (cum_log_probs == nullptr && output_log_probs == nullptr) {
            for (int i = 0; i < k; ++i) {
                s_max = fmaxf(s_max, topk_vals[i]);
            }
        }

        // --- CHECK ---
        // for (int i = 0; i < vocab_size_padded; i++) {
        //     std::cout << topk_vals[i] << " ";
        // }
        // std::cout << std::endl;
        // -------------

        // Step 3: Softmax 处理并归一化
        // 在 Top-k 采样的代码中，归一化是隐含在随机数采样过程中完成的
        float s_sum = 0.0f;          // 表示 topk 选择的概率和
        for (int i = 0; i < k; ++i) {
        // when cum_log_probs are computed, topk_vals (logits_buf_) are already pre-processed by softmax_kernel
            if (cum_log_probs == nullptr && output_log_probs == nullptr) {
                topk_vals[i] = expf(topk_vals[i] - s_max);              // Numerically stable softmax
            }
            s_sum += topk_vals[i];
        }

        // std::cout << s_sum << std::endl;

        // Step 4: 生成随机数，进行 Top-k 采样
        float rand_num = rand_float() * prob_threshold * s_sum;
        
        // std::cout << rand_num << std::endl;

        // 通过逐渐减去概率值直到随机数 rand_num 变为负数时停止，实现了概率分布下的采样
        for (int i = 0; i < k; i++){
            rand_num -= topk_vals[i];
            if (rand_num <= 0.0f || i == k - 1) {
                
                // std::cout << "topk:" << topk_vals[i] << std::endl;

                ids[batch_id] = topk_indices[i];
                
                // 计算累积 log 概率 输出 log 概率
                if (cum_log_probs != NULL || output_log_probs != NULL) {
                    float log_prob = logf(topk_vals[i]);
                    if (cum_log_probs != NULL) {
                        cum_log_probs[batch_id] += log_prob;
                    }
                    if (output_log_probs != NULL) {
                        // log_prob = log P(i | i is in top-k) = log(exp_logit / s_sum)
                        output_log_probs[batch_id] = log_prob - logf(s_sum);
                    }
                }
                break;
            }
        }

        // std::cout << " - " << cum_log_probs[batch_id] << " " << output_log_probs[batch_id] << std::endl;

        // Step 5: 更新序列长度和 finished 状态
        if (sequence_lengths != NULL && finished != NULL) {
            sequence_lengths[batch_id] = finished[batch_id] ? sequence_lengths[batch_id] : sequence_lengths[batch_id] + 1;
            finished[batch_id] = (ids[batch_id] == end_ids[batch_id]);
        }
    }
}

void cpuInvokeStopWordsCriterion(
                           const int* output_ids,
                           const int* stop_words,
                           bool* finished,
                           int stop_words_len,
                           int batch_size,
                           int step
) {
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            const int* base_stop_words = stop_words + batch_idx * 2 * stop_words_len;
            const int* base_offsets = base_stop_words + stop_words_len;
            for (int id = 0; id < stop_words_len; ++id) {
                if (base_offsets[id] < 0) {
                    continue;
                }
                const int item_end = base_offsets[id];
                const int item_start = (id > 0) ? base_offsets[id - 1] : 0;
                const int item_size = item_end - item_start;

                bool should_stop = false;
                if (step + 1 >= item_size) {
                    should_stop = true;
                    for (int token_idx = item_size - 1; token_idx >= 0; --token_idx) {
                        const int previous_token = output_ids[(step - (item_size - 1) + token_idx) * batch_size + batch_idx];
                        if (previous_token != base_stop_words[item_start + token_idx]) {
                            should_stop = false;
                            break;
                        }
                    }
                }
                if (should_stop) {
                    finished[batch_idx] = true;
                }
            }
    }
}


void cpuInvokeLengthCriterion(
    bool* finished,
    bool* should_stop,
    int* finished_sum,
    int* sequence_limit_length,
    int batch_size,
    int step
) {
    int finished_count = 0;
    for (int batch_index = 0; batch_index < batch_size; batch_index++){
        finished[batch_index] |= step >= sequence_limit_length[batch_index];
        finished_count += finished[batch_index] ? 1 : 0;
    }
    finished_sum[0] = finished_count;
    should_stop[0] =  finished_count == batch_size;
}


// CHECK
void print_logits(float* logits, int len, std::string name) {
    std::cout << std::left << std::setw(25) << std::setfill(' ') << name;
    for (int i=0; i < len; i++) {
        std::cout << std::left << std::setw(10) << logits[i] << " ";
    }
    std::cout << std::endl;
}

template <typename T>
void print(T* input, int len, std::string name) {
    std::cout << std::left << std::setw(15) << std::setfill(' ') << name;
    for (int i=0; i < len; i++) {
        std::cout << std::left << std::setw(5) << input[i] << " ";
    }
    std::cout << std::endl;
}



// ALL BATCH COMPETE
void cpuCustomTransformerDynamicDecoder(

    float* logits,                      // [batch_size * vocab_size_padded]      vocab logits
    int* output_ids,                    // [batch_size * sequence_limit_length]
        
    int step,                           // [1]                     解码过程的 step 从 max_input_length 开始递增，step 每自增1，代表解码一次
    int batch_size,                     // [1]
    int vocab_size,                     // [1]                     词表大小
    int vocab_size_padded,              // [1]                     填充后的词表大小
    int *end_ids,                       // [batch_size]            结束词 id
    
    bool share_words,                   // [1]                     bad_word 是否不同 batch 之间共享
    int* bad_words_list,                // [bad_words_len * 2]     bad_words
    int bad_words_len,                  // bad_words 的总长度
    
    float* embedding_bias,              // [vocab_size]            bias
    float* temperature,                 // [1] or [batch_size]     mul
    int temperature_size,               // [1]                     决定是否不同 batch 不同的 temperature
    float repetition_penalty,           // [1]                     惩罚数据
    RepetitionPenaltyType penalty_type, // [1]                     决定哪种惩罚方式

    int* sequence_lengths,              // [batch_size]            序列长度
    int* sequence_limit_length,         // [batch_size]            序列的最大长度限制
    int max_input_length,               // [1]                     输入序列的最大长度
    int* input_length,                  // [batch_size]            输入序列的长度
    int min_length,                     // [1]                     num_gen_tokens 最小总长度

    int max_top_k,                      // [1]                     top_k k
    int* top_ks,                        // [batch_size]            each batch k
    float top_p,                        // [1]                     top_p
    float* top_ps,                      // [batch_size]            each batch p
    float* cum_log_probs,               // [batch_size]            累积对数概率
    float* output_log_probs,            // [batch_size * (sequence_limit_length - max_input_length)]     所选 token 相对 Top-k 集的概率 

    int stop_words_len,                 // [1]                     停用词总长度
    int* stop_words_list,               // [stop_words_len * 2]    停用词列表
    
    bool* finished,                     // [batch_size]            指示该 batch 是否结束
    int* finished_sum,                  // [1]                     结束 batch 的个数和
    bool* should_stop,                  // [1]                     标志结束 所有 batch 是否完成 
    
    bool* skip_decode                   // [batch_size]            batch 是否跳过 decoder
){

    print_logits(logits, vocab_size_padded, "INPUT");

    if (bad_words_list != nullptr){
        cpuInvokeBanBadWords(
            logits,
            output_ids,
            bad_words_list,
            bad_words_len,
            share_words,
            batch_size,
            vocab_size_padded,
            step
        );
    }

    print_logits(logits, vocab_size_padded, "AFTER BanWords");


    if (embedding_bias != nullptr || temperature != nullptr){
        cpuInvokeTemperaturePenalty(
            logits,
            embedding_bias,
            temperature,
            temperature_size,
            batch_size,
            vocab_size,
            vocab_size_padded
        );
    }

    print_logits(logits, vocab_size_padded, "AFTER Temperature");

    if (step > 0 && penalty_type != RepetitionPenaltyType::None) {
        cpuInvokeRepetitionPenalty(
            logits,
            output_ids,
            input_length,
            max_input_length,
            batch_size,
            vocab_size,
            vocab_size_padded,
            step,
            penalty_type,
            repetition_penalty
        );
    }

    print_logits(logits, vocab_size_padded, "AFTER Repetition");

    if (step - max_input_length < min_length) {
        cpuInvokeMinLengthPenalty(
            logits,
            min_length,
            end_ids,
            sequence_lengths,
            max_input_length,
            batch_size,
            vocab_size_padded
        );
    }

    print_logits(logits, vocab_size_padded, "AFTER MinLength");

    cpuInvokeAddBiasEndMask(
        logits,
        nullptr,
        end_ids,
        finished,
        batch_size,
        vocab_size,
        vocab_size_padded
    );

    print_logits(logits, vocab_size_padded, "AFTER AddBiasEndMask");

    if (cum_log_probs != nullptr || output_log_probs != nullptr) {
        cpuInvokeAddBiasSoftMax(
            logits,
            nullptr,
            end_ids,
            finished,
            batch_size,
            vocab_size,
            vocab_size_padded
        );

        print_logits(logits, vocab_size_padded, "AFTER AddBiasSoftMax");
    }
    
    cpuInvokeBatchTopKSampling(
        logits,
        output_ids + step * batch_size,
        sequence_lengths,
        end_ids,
        finished,
        max_top_k,
        top_ks,
        top_p,
        top_ps,
        batch_size,
        vocab_size,vocab_size_padded,
        cum_log_probs,
        output_log_probs + (step - max_input_length) * batch_size,
        skip_decode
    );


    print_logits(logits, vocab_size_padded, "AFTER TopKSampling");
    print(output_ids, step+1, "ouput_ids");
    print(finished, batch_size, "finished_1");

    if (stop_words_list != nullptr){
        cpuInvokeStopWordsCriterion(
            output_ids,
            stop_words_list,
            finished,
            stop_words_len,
            batch_size,
            step
        );
    }
    print(finished, batch_size, "finished_2");
    print(should_stop, 1, "should_stop1");


    if (sequence_lengths != nullptr){
        cpuInvokeLengthCriterion(
            finished,
            should_stop,
            finished_sum,
            sequence_limit_length,
            batch_size,
            step
        );
    }
    print(should_stop, 1, "should_stop1");
}



int main(){
    // 1. 定义常量
    int batch_size = 1;                
    int vocab_size = 5;          
    int vocab_size_padded = 5;
    int max_input_length = 4;
    int end_ids[] = {4}; 
    bool share_words = true; 
    int bad_words_list[] = {1, 2, 2, -1};
    int bad_words_len = 1;
    float embedding_bias[] = {0.1, 0.1, 0.1, 0.1, 0.1};
    float temperature = 0.5;
    int temperature_size = 1;
    float repetition_penalty = 0.1; 
    RepetitionPenaltyType penalty_type = RepetitionPenaltyType::Additive; 
    int min_length = 3;
    int max_top_k = 3;
    int top_ks[] = {2};
    float top_p = 0.9;
    float top_ps[] = {0.8};
    int stop_words_list[] = {0, 1};
    int stop_words_len = 1;
    
    // 2. 输入数据初始化
    float logits[1][5] = {
        {0.2, 0.5, 0.1, 0.15, 0.05}
    };
    int output_ids[10] = {2, 1, 3, 1};
    int step = max_input_length;
    int sequence_lengths[1] = {max_input_length};
    int sequence_limit_length[1] = {10};
    int input_length[1] = {max_input_length};
    bool finished[1] = {false};
    int finished_sum[1] = {0};
    bool should_stop[1] = {false};
    bool skip_decode[1] = {false};
    float cum_log_probs[1] = {0.0};
    float output_log_probs[10];

    // 3. 调用函数
    cpuCustomTransformerDynamicDecoder(
        &logits[0][0],          // logits
        output_ids,                     // output_ids
        step,                           // step
        batch_size,                     // batch_size
        vocab_size,                     // vocab_size
        vocab_size_padded,              // vocab_size_padded
        end_ids,                        // end_ids
        share_words,                    // share_words
        bad_words_list,                 // bad_words_list
        bad_words_len,                  // bad_words_len
        embedding_bias,                 // embedding_bias
        &temperature,                   // temperature
        temperature_size,               // temperature_size
        repetition_penalty,             // repetition_penalty
        penalty_type,                   // penalty_type
        sequence_lengths,               // sequence_lengths
        sequence_limit_length,          // sequence_limit_length
        max_input_length,               // max_input_length
        input_length,                   // input_length
        min_length,                     // min_length
        max_top_k,                      // max_top_k
        top_ks,                         // top_ks
        top_p,                          // top_p
        top_ps,                         // top_ps
        cum_log_probs,                  // cum_log_probs
        output_log_probs,               // output_log_probs
        stop_words_len,                 // stop_words_len
        stop_words_list,                // stop_words_list
        finished,                       // finished
        finished_sum,                   // finished_sum
        should_stop,                    // should_stop
        skip_decode                     // skip_decode
    );
    
        return 0;
}