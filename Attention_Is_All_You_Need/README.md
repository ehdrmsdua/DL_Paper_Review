Abstract
- 기존 Sequence transduction model들은 Encoder와 Decoder를 포함한 RNN과 CNN을 기반으로 한다.

- 본 논문에서는 이러한 RNN과 CNN을 배제하고 오로지 Attention 메커니즘만을 기반으로 한 Transformer를 제안한다.

- 이는 기존 모델보다 병렬화가 쉬워 학습 시간을 크게 단축시키고, 성능 또한 우수하였다.



1. Introduction
- 기존에는 RNN, LSTM, GRU 등의 순환 모델이 성능이 좋고, 많이 이용되었다.

- 위와 같은 순환 모델은 $h_{t}$가 이전 hidden state인 $h_{t-1}$과 현재 $t$에서의 입력을 기반으로 생성되는 순차적 특성을 가진다.

- 해당 특성은 병렬화를 불가하게 하며, 시퀀스의 길이가 길어질 수록 모델 성능을 저하시킨다.

- 기존 Attention Mechanism 또한 부분적으로 RNN을 활용하여 이러한 제약에서 자유롭지 못했다.

- 본 연구에서 제안하는 Transformer 모델은 순환 구조를 제거해, 병렬화를 가능하게 하고, 성능 또한 크게 향상시켰다.



2. Background
- Sequential computation 감소를 목표로 하는 기존 Extended Neural GPU, ByteNet , ConvS2S는 CNN을 기반으로 함

- 해당 모델들은 Hidden representation을 병렬로 계산할 수 있지만, 연산 수가 위치간 거리에 따라 증가한다.

(ConvS2S는 선형적 증가, ByteNet는 로그 형태로 증가)

- Transformer는 이러한 연산 수를 상수 수준으로 줄임.



3. Model Architecture

위 그림은 Transformer의 architecture로 Encoder(왼쪽)와 Decoder(오른쪽)가 나타나 있다.

- Encoder는 Input Sequence (((x_{1}, ... x_{n})))를 연속적인 표현(((z_{1}, ... z_{n})))의 Sequence로 매핑한다.

- Decoder는 해당 $z$를 기반으로 Output Sequence(((y_{1}, ... y_{m})))를 순차적으로 생성한다.

- Model은 각 단계에서 자가 회귀적(Auto-Regressive)으로 동작하며, 이전에 생성된 심볼을 다음 입력으로 사용한다.



3.1 Encoder and Decoder Stacks



1. Encoder

- Encoder는 $N = 6$의 동일한 Stacks로 구성되며, 두 개의 하위 Layers로 구성되어 있다.

- Encoder에서는 Multi-Head Attention(Self-Attention) 메커니즘과 Fully Connected Feed-Forward Network가 사용된다.

- 하위 Layers 각각에는 Residual Connection과 Layer Normalization이 진행된다.

(*Residual Connection은 Transformer의 Back Propagation 과정에서 Activation Function로 인해 기울기가 지수적으로 감소하는 Gradient Vanishing Problem을 완화하기 위해 적용한 것으로 보인다.)

- 앞서 하위 Layers에서 적용되는 Residual Connection과 Layer Normalization은 다음과 같은 수식으로 나타낼 수 있다.

\[\text{LayerNorm}(x + \text{Sublayer}(x))\]

- Residual Connection을 위해선 Layer Output의 Dimension이 같아야 하므로, 모든 하위 Layers와 Embedding Layers의 출력 Dimension은 ((d_{\text{model}} = 512)) 로 고정하였다.



2. Decoder

- Decoder는 ((N=6))의 동일한 Stacks로 구성되며, Encoder의 하위 Layers와 세번째 하위 Layers를 추가로 포함한다.

- 해당 Layer는 마지막 Encoder Stack의 출력과의 Multi-Head Attention을 수행한다.

- 하위 Layers 각각에는 Residual Connection과 Layer Normalization이 진행된다.

- Decoder의 Self-Attention에서는 이후 위치의 정보를 유출시키지 않기 위해 해당 정보들을 Masking($ -∞$로 변환)한다.

(* -∞와 같이 큰 음수 값으로 변환하면 SoftMax 과정에서 확률이 0에 가까워진다. ← 참조를 방지)

- Decoder는 Output Embedding을 한 위치 뒤로 이동시켜 입력으로 사용한다.

(시작: <SOS> =  Start Of Sequence, 종료: <EOS> = End Of Sequence)



3.2 Attention


- Queries, Keys, Values, Outputs는 모두 Vector 형태이다.

- Attention 함수의 출력은 Value의 가중합(Weighted Sum)으로 계산된다.

- 가중치는 Query와 Key의 호환성 함수(Compatibility function)을 통해 계산된다.

(*논문에서는 $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$ 형태로 사용)


3.2.1 Scaled Dot-Product Attention

-  (Figure 2)의 Attention 방식을 Scaled Dot-Product Attention이라고 한다.

- 입력 값은 Dimension이 ((d_{k}))인 Query와 Key, Dimension이 ((d_{v}))인 Value로 구성된다.

- 계산과정은 다음과 같다.

 1.Query와 모든 Key의 내적을 계산한다.

 2.값을 $\sqrt{d_k}$로 나누어 스케일링한다.

 3. SoftMax를 취하여 Value의 가중치를 구한다.

- 실제로는 Query는 Q, Key와 Value는 K,V라는 행렬로 묶어 동시에 계산을 진행한다

- 이를 수식으로 나타내면 아래와 같다.

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

- 일반적으로 Attention 함수에는 Additive Function과 Dot-Product Function이 자주 사용된다.

- Dot-Product Function은 $\sqrt{d_k}$ Scaling을 제외하면 사용하는 메커니즘과 동일하다

- Additive Function은 Single Hidden Layer를 가진 Feed-Forward Network를 사용한다.

- 이론적으론 복잡도가 비슷하나, Dot-Product를 이용한 계산이 더 빠르고 효율적이다.

- ${d_k}$가 작은 경우, 두 메커니즘의 성능이 비슷하지만, ${d_k}$가 클 때, Dot-Product Function이 상대적으로 낮은 성능을 보인다.

- 이는 ${d_k}$가 클 때, Dot-Product 값의 분산이 커져 SoftMax 과정에서 Gradient 값이 극단적으로 계산된 것으로 보인다.

- 이 문제를 해결하기 위해 값을 $\sqrt{d_k}$로 나누어 스케일링 하였다.



3.2.2 Multi-Head Attention

- Query, Key, Value 각각을 ((h))번 Linear Projection하여 Dimension이 각각 ${d_k}$, ${d_k}$, ${d_v}$가 되도록 변환할 수 있다.

- Projection 된 Query, Key, Value에 대해 병렬적으로 Attention 함수를 적용하여, Dimension이 ${d_v}$인 Outputs을 생성한다.

- Outputs들은 Concat된 후 다시 Projection되어 최종 Output을 출력한다. (위 Figure 2 right)

- 이와 같은 작업은 모델이 데이터를 다양한 관점에서 바라보게 하여, 더 다양성 있는 정보들을 추출할 수 있도록 한다.

-이를 수식으로 나타내면, 아래와 같다.

\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O, \quad
\text{where } \text{head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)
\]

-Projection 과정에서 행렬은 아래와 같이 정의된다.

\[
W^Q_i \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad 
W^K_i \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad 
W^V_i \in \mathbb{R}^{d_{\text{model}} \times d_v}, \quad 
W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}.
\]

(*$W_ {\text{Q,K,V}} $는 Projection을 위해 해당하는 Dimension을 가지고, 이를 다시 Projection하기 위해, $W_O$는 $h$$d_v$ x $ d_{\text{model}} $로 정의됨을 알 수 있다.)

- 논문에서는 $h=8$의 병렬 Attention Layer를 사용하여 각 head에 대해 $d_k = d_v = \frac{d_{\text{model}}}{h} = 64$를 사용하였다.



3.2.2 Applications fo Attention in our Model

- Transformer에서는 Multi-Head Attention을 세 가지 방식으로 사용한다.



- Encoder-Decoder Attention

Query는 이전 Decoder 층에서 가져오며, Key와 Value는 Encoder의 출력에서 가져온다.
이는 입력 Sequence의 모든 위치를 참조할 수 있도록 한다.
- Encoder의 Self-Attention

Query, Key, Value는 모두 Encoder의 이전 층에서 가져온다.
Encdoer의 각 입력값은 다른 모든 입력값을 참조할 수 있다.
- Decoder의 Self-Attention

Query, Key, Value는 모두 Decoder의 이전 층에서 가져온다.
Decoder의 각 입력값은 이전 위치와 해당 위치의 값만 참조할 수 있다.
자가 회귀적 특성을 유지하기 위해, Scaled Dot-Product Attention에서 참조 불가 값들은 $-∞$로 Masking한다
3.3 Position-wise Feed-Forward Networks



- Encoder와 Decoder의 각 층에는 Fully Connected Feed-Forward Network가 포함되어 있다.

- 이 네트워크는 Sequence 모든 위치에 에 독립적으로 적용되며, 두 개의 선형 변환과 ReLU activation function으로 구성되어있다.

- 이를 수식으로 나타내면 아래와 같다.

\[\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2\]

- 선형 변환은 각 Layer 마다 다른 매개변수($W_1$,$W_2$ ,$b_1$,$b_2$ )를 사용한다.

- 입력과 출력의 Dimension은 $d_{\text{model}} = 512$이고, 내부 Layer의 Dimension은 $d_{\text{ff}} = 2048$이다.

(*더욱 복잡한 표현들을 학습할 수 있게 한다.)



3.4 Embedding and Softmax



- Embedding: 학습된 Embedding을 사용하여, 입출력 Token을 Dimension이 $d_{\text{model}}$인 Vector로 변환한다.

- Decoder의 출력은 학습된 선형변환과 Softmax를 사용하여 다음 Token에 대한 확률로 변환한다.

- Model에서는 Outputs Embedding, Inputs Embedding Layers와 SoftMax 이전의 선형변환에서 Embedding Vector의 Weight Matrix는 공유된다.

- Embedding Layer에서는 이러한 Weight Matrix에 $\sqrt{d_{\text{model}}}$을 곱해 추가적인 Scaling을 진행한다.

(*Embedding에서 $W_{\text{embed}} \sim \mathcal{N}\left(0, \frac{1}{d_{\text{model}}}\right)$에서 시작하면 되는게 아닌가 싶어 오래 생각해봤는데, 이 같은 방법은 학습 중에 해당 Scale이 유지되지 않을 가능성이 있어서 직접적으로 곱해주는 것이라고 결론내렸다. 실험적으로도 본 논문과 같은 방법이 더 좋은 성능을 보였다고 한다.(GPT가 그랬다!))

-아래는 Layer 유형에 따른 복잡도, 연산 횟수, 최대 경로의 길이이다.




3.5 Positional Encoding



- Transformer에는 Sequence의 순서를 모델이 알 수 있도록 Embedding 이후 Positional Encoding을 더하는 방식을 채택했다.

- Transformer는 이와 같은 위치 인코딩을 Sine 함수와 Cosine 함수로 나타내었다. 이때 결과 Vector의 Dimension은 $d_{model}$이다.

- 사용된 수식은 아래와 같다.

\[
\text{PE}(\text{pos}, 2i) = \sin\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
\]

\[
\text{PE}(\text{pos}, 2i+1) = \cos\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
\]



- 여기서 $pos$는 Token의 위치, $i$는 Vector의 차원이다.

- 짝수 차원은 Sine 함수를, 홀수 차원은 Cosine 함수를 활용한다.

- 학습이 가능한 위치 Embedding 방식도 있지만 실험 결과, 두 방식이 거의 동일한 결과를 나타내었다.

- 허나 Train 과정에서 없던 더 긴 Sequence의 길이에서 해당 방식이 더 일반화 성능이 우수할 것이라 생각하여 이 방식을 채택했다.



 4. Why Self-Attention
- Self-Attention Layer는 모든 Sequence 위치의 Path Length가 $O(1)$ 수준에서 계산되고, Recurrent Layer는 $O(n)$, Convolution Layer는 Kernel의 크기 $k$에 따라 $O(log_{k}(n))$이나 $O(n/k)$가 요구된다.

- 상수 수준의 Path Length는 Long-range Dependency를 학습하는데에 있어 더 안정적이다.

- Self-Attention의 계산 복잡도는 Sequence의 길이 $n$보다 Representation Dimension $d$가 작을 때 Recurrent Layer보다 더 효율적인데, 이 같은 특성이 최근 개발된 Tokenizer인 Word-piece와 Byte-pair에 부합한다.

- Sequence를 순차적으로 처리할 필요가 없어, 병렬화가 가능하다.



5. Training 
* 4개 목차 통합했으므로 세부 내용은 논문 원문 참고



- WMT 2014 English-German, English-French 데이터셋(각 450만, 3,600만 개 문장으로 구성)을 각각 BPE,Wordpiece로 토큰화

- NVDIA P100 GPU 8개 사용하여 Base,Big으로 나누어 각각 10만 step(12시간 소요), 30만 step(3.5일 소요)으로 학습

- Adam Optimizer를 사용하였고, Parameters는 $\beta_1 = 0.9, \quad \beta_2 = 0.98, \quad \epsilon = 10^{-9}$

- 이 때 Learning Rate의 변화는 Warmup_steps(4,000) 동안 선형적 증가, 이후 Step의 제곱근에 반비례하며 감소

- Residual Dropout($P_{drop}=0.1$)과 Label Smoothing($\epsilon_{\text{ls}} = 0.1$)을 적용

- 학습 결과, Transformer Base, Big 모델은 BLEU (27.3,38.1),(28.4,41.8) Cost $3.3 \cdot 10^{18} \, \text{FLOPs}$, $2.3 \cdot 10^{19} \, \text{FLOPs}$의 결과를 보임

- 이는 타 Model(ByteNet, MoE, ConvS2S 등)보다 적은 학습 비용으로 더 높은 BLEU 점수를 기록한 것이다.

- 자세한 사항은 아래 표를 참고.


6. Reuslts
* 3개 목차 통합했으므로 세부 내용은 논문 원문 참고



- English-German의 Transformer(big,base) 모델은 더 적은 cost로 이전 최고 성능(Ensemble 포함) BLEU를 초과한 결과를 나타냄.

- English-French의 Transformer(big,base) 모델은 1/4 cost로 이전 최고 성능(단일 model) BLEU를 초과한 결과를 나타냄.

- Transformer를 변형 하였을 때 결과는 아래와 같다.

1. Multi-Head Attention의 / Head =1 → BLEU 0.9 감소, Head = 과하게 많음 → 성능 감소

2. Attention Key의 Dimension $d_{k}$ → 품질 저하

3. 모델의 크기 증가 → 성능 증가, Drop-out 적용 → 성능 증가

4. Sine,Cosine Positional Embedding에서 학습 Positional Embedding으로 변경 → 동일한 성능.

-자세한 사항은 아래 표를 참고.




- Wall Street Journal(이하 WSJ)에서 RNN과 Sec2Sec 모델은 성능이 기대 이하였다.

- 4-Layers Transformer를 사용하여 WSJ에서 약 4만 개 학습 문장, Semi-Supervised에서 1,700만 개 문장을 포함, Parsing을 위해 BerkeleyParser라는 데이터 셋 추가 사용

- 일부 Hyper Parameters만 실험을 통해 조정하고 실험 진행, RNN과 Sec2Sec보다 학습 데이터가 적지만 더 좋은 성능.

- 작업별 튜닝 없이도 성능이 우수했음, RNN Grammer(아래 표 [8])를 제외한 모든 모델보다 더 우수한 성능을 보임.

- 자세한 사항은 아래 표를 참고


7. Conclusion
- Transformer의 핵심은 기존 Recurrent Layer를 Multi-head Self-Attention으로 바꾼 것이다.

- 작업에서의 효율과 성능은 실험을 통해 검증 되었다.

- 향후 텍스트 외에도 이미지, 오디오 등의 출력을 효율적으로 처리하기 위한 방법도 연구할 예정이다.
