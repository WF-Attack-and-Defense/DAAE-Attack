# DAAE: Defense-Aware Website Fingerprinting via Learning Defense-Induced Traffic Transformations

## Introduction

The Tor network employs layered encryption and multi-hop routing through a sequence of relays to protect users' identities and browsing activities. Despite these anonymity mechanisms, website fingerprinting (WF) attacks remain a persistent threat to Tor users' privacy. WF attacks exploit traffic characteristics that remain observable even when application-layer contents are encrypted, enabling an adversary to infer the websites visited by a user. In particular, encrypted Tor traffic retains side-channel information such as packet timing, burst patterns, and packet directions. By learning distinctive patterns in these features, an adversary can associate an observed traffic trace with a candidate website without directly accessing the user's encrypted content.

The development of deep learning (DL) techniques has significantly enhanced the effectiveness of WF attacks. These approaches can automatically learn distinctive patterns from encrypted traffic traces and achieve high classification performance under various experimental settings. This progress has consequently motivated extensive research on defenses against WF attacks. Existing defenses generally aim to disrupt the traffic characteristics exploited by WF classifiers. For example, adaptive padding-based techniques obscure timing and burst information by adaptively or randomly injecting dummy packets. Adversarial perturbation techniques instead construct carefully designed perturbations that alter traffic traces in ways intended to mislead DL-based WF classifiers. Regularization-based approaches transform traffic into more regular transmission patterns, reducing the distinguishability of traces associated with different websites.

Recently, adversarial training (AT)-enhanced WF attacks  have been introduced to defend against the previously proposed defense mechanisms. However, the state-of-the-art (SOTA) defense methods have proven to be strong at defending against AT-enhanced WF attacks. For instance, Regulator regularizes the size and shape of packet `surges' that frequently occur in download traffic, masking potentially revealing features. In this paper, `surge' is broadly defined as a large number of packets sent over a short period of time. Palette first clusters website traffic traces with high-similarity traffic patterns offline and then applies traffic regularization online to enforce real-time traces into a uniform traffic pattern. Considering that Palette is limited to its predefined dataset for clustering.  Adaptive Tamaraw proposes an adaptive WF defense framework with dynamic clustering to adjust defense parameters in real time.

To undermine the effectiveness of WF defenses that are robust against AT-based attacks, we analyze existing defense mechanisms and find that most induce stable, learnable mappings in traffic patterns, which can be exploited by *defense-aware* attacks. We utilize the mapping method to functionalize the relationship between the undefended Tor traffic traces and the defended Tor traffic traces. We found that the existing defense mechanisms usually perform a one-to-one mapping or many-to-one mapping. Through analyzing the mapping strategy, we consider introducing an advanced WF attack mechanism that enhances the WF attack performance. Specifically, we assume a strong WF adversary who is aware that a defense is deployed on Tor but does not know the specific defense mechanism being used (black-box). The adversary can train a Defense-Aware Autoencoder (DAAE) model to learn defense-induced perturbation patterns using paired original Tor website traces and defense-perturbed traces. Then, use the trained model to mark and remove potential injected dummy packets from defense traces. 

In this paper, we propose DAAE, a novel framework for removing dummy packets from defense traces. Specifically, DAAE first employs longest common subsequence (LCS) alignment on paired defended and undefended traces to generate an LCS Keep Mask, which labels each packet as either an original packet (marked as 1) or a dummy packet (marked as 0). The extracted LCS packet sequences are also stored in an LCS Bank, which serves as a class-specific feature repository. Next, the LCS Bank and the corresponding LCS Keep Masks are used to train a few-shot-based DAAE model, FS-DAAE, which consists of a one-dimensional U-Net decoder and a four-stage encoder built from stride-2 one-dimensional convolutional (Conv1D) layers. Once trained, FS-DAAE generates mask logits for previously unseen defended traces, enabling the identification and removal of dummy packets. The resulting recovered traces are then fed into deep learning-based WF attack models, including DF, Var-CNN, and NetCLR, to improve attack effectiveness. Specifically, our work addresses the following two challenges.

*First, how can we achieve robust and effective WF attacks across diverse defense mechanisms, and how can we learn defense-induced packet injection patterns without access to the internal implementations of WF defenses?* Existing WF defenses employ diverse strategies, including traffic obfuscation, traffic regularization, adversarial perturbation, and website morphing. Our analysis reveals that these defenses induce learnable mappings between undefended and defended Tor traffic traces, typically exhibiting one-to-one or many-to-one relationships. To exploit these mappings, we design a Defense-Aware Autoencoder (DAAE) that learns to distinguish original packets from defense-injected dummy packets. Specifically, we first align paired undefended and defended traces using longest common subsequence (LCS) matching to construct packet-level masks, where original and dummy packets are labeled as 1 and 0, respectively. We then store the aligned packet sequences in a class-specific LCS Bank and use the sequences and masks to train a few-shot DAAE model (FS-DAAE), comprising a stride-2 Conv1D encoder and a one-dimensional U-Net decoder. Given a previously unseen defended trace, FS-DAAE predicts a packet-level mask to identify and remove dummy packets, allowing existing WF classifiers to operate on the recovered traces. This approach enables defense-aware attacks without requiring access to the defense implementation.

*Second, how can we accurately identify and remove dummy packets while preserving original packets?* Since dummy packets are interleaved with original packets and may exhibit similar direction patterns, packet-level classification can mistakenly remove original packets or retain dummy packets, compromising trace recovery and subsequent WF attacks. To address this challenge, DAAE leverages longest common subsequence (LCS) alignment between paired defended and undefended traces to generate packet-level keep masks, providing supervision for distinguishing original packets from injected dummy packets. We store the aligned sequences in a persistent, class-specific LCS Bank and train FS-DAAE to combine support-trace features with query-trace representations through support–query fusion, feature-wise linear modulation, and cross-attention. The U-Net decoder then produces packet-level keep logits for each defended query. During inference, rather than independently thresholding these logits, DAAE employs insertion-only Viterbi decoding to find a high-scoring keep mask under the constraint that the original trace must be a subsequence of the defended trace. Finally, it retains the selected packets in their original order and removes the rest, without accessing the query's original trace or any validation/test samples during recovery.

Our evaluation across two widely used datasets and seven state-of-the-art (SOTA) WF defenses demonstrates that DAAE substantially improves attack effectiveness, particularly against defenses designed to resist adversarial training (AT)-based attacks. On the DF dataset, DAAE achieves classification accuracies of up to 97.85\%, 98.47\%, and 97.83\% for DF, Var-CNN, and NetCLR, respectively. On the GTT23 dataset, DAAE achieves accuracies of up to 86.03\%, 83.65\%, and 86.31\% for the three attack models, respectively. Notably, against RegulaTor, Palette, and Adaptive Tamaraw, DAAE consistently outperforms both attacks without AT and attacks with AT, demonstrating its effectiveness in recovering discriminative traffic patterns under challenging defense settings.

 - We analyze existing WF defense mechanisms and identify stable, learnable mappings between defended and undefended traffic traces. Our findings reveal that these mappings can be exploited by defense-aware attacks, even against defenses designed to resist AT-based WF attacks.
 - We propose DAAE, a novel framework that combines LCS alignment and an FS-DAAE model to identify and remove dummy packets from defended traces. By leveraging class-specific packet sequence features and learned packet masks, DAAE aims to recover the underlying traffic patterns while reducing the need for extensive unlabeled traces.
 - We conduct a comprehensive evaluation of DAAE on two widely used public datasets and against seven SOTA defense mechanisms, demonstrating its effectiveness in removing dummy packets and improving WF attack performance across diverse defense settings.

## Threat Model and Problem Statement

### Threat Model

![Figure 1: Overview of the Tor network and Threat Model.](./images/threatmodel.png)
Figure 1: Overview of the Tor network and Threat Model. 

*Attack Scenario*： We consider the standard local passive adversary that monitors the communication network traffic between the Tor client and the entry guard but cannot modify traffic packets. As shown in Figure 1, potential adversaries include eavesdroppers on the client's local network, Internet service providers (ISPs), autonomous systems (ASes), and malicious Tor guard nodes. Although Tor encrypts traffic contents, these adversaries can observe traffic characteristics that may reveal information about the websites visited by the client.
To defend against WF attacks, a WF defense can be deployed through a pluggable transport (PT) bridge, which operates between the Tor client and the entry guard and transforms observable traffic patterns before they traverse the monitored network path.

*Attacker Capability*: We consider a strong adversary with access to defended traffic traces from monitored webpages collected under conditions comparable to those experienced by a normal user. We assume that the adversary can operate as a Tor user and collect both the original Tor traffic at the client side (① in Figure 1) and the corresponding defended traffic transmitted through the PT client and PT server (② in Figure 1). Additionally, the adversary can passively capture the victim's network traffic between the victim's device and the entry guard. Using the captured network traffic traces, the adversary can train WF classifiers under two attack strategies: *adversarial training*-based WF attacks and *DAAE*-based WF attacks. The former incorporates defended traffic into the training process to improve classification performance against the deployed defense, while the latter leverages a *DAAE*-based approach to learn the dummy packet injection mechanism without knowledge of the defense algorithm, thereby facilitating website identification despite traffic transformations. By considering both strategies, we evaluate the effectiveness of*DAAE*-based WF attacks against WF defenses in the presence of defense-induced changes in traffic patterns.

*Attacker Knowledge*: The *DAAE* does not require prior knowledge of the destination webpage. Thus, *DAAE* can be applied to arbitrary browsing webpages rather than a pre-configured set of static candidate traces. Additionally, PTs are independently deployed by users. Therefore, the defense for different users does not rely on the same predefined payloads or destination-specific configurations. In this setting, *DAAE* can avoid introducing stable, webpage-dependent artifacts that an adaptive adversary can learn.

*Closed- and Open-World Scenarios*: WF attacks are commonly evaluated under closed-world and open-world scenarios. In the closed-world scenario, the client is assumed to visit only a predefined set of websites, referred to as *monitored* websites. The attacker has access to traffic traces collected from these websites and uses them to train a classifier to identify which monitored website the client visits. In contrast, the open-world scenario reflects a more realistic setting in which the client may visit either monitored websites or a much larger set of *unmonitored* websites. Since the attacker can obtain training traces for a subset of unmonitored websites, the goal is to determine whether the client's traffic corresponds to one of the monitored websites.

### Problem Formulation

Tor network traffic can be represented as a time-ordered sequence of packets. Since packet timing is sensitive to network conditions and user location, we focus exclusively on packet directions in this work. Formally, each traffic trace corresponding to the loading of a webpage is represented as:

$$
w = [d_1, d_2, \ldots, d_N]
$$

Here, $N$ is the total number of website packets observed during the session. $d_i \in \{+1, -1\}$ denotes the direction of the $i$-th packet. In particular, $+1$ denotes an outgoing packet, and $-1$ denotes an incoming packet.

Let $\mathcal{W}$ denote the monitored websites and $\mathcal{X}$ denote the space of observable traffic. An adversary aims to learn a classification model:

$$
h: \mathcal{X} \rightarrow \mathcal{W}
$$

that predicts the specific website from an observed trace. The WF defense system can be regarded as a stochastic transformation mechanism:

$$
\mathcal{D}: \mathcal{X}' \rightarrow \mathcal{D}(\mathcal{X})
$$

where $\mathcal{D}(\mathcal{X})$ denotes a distribution over defended traces.

$\mathcal{R} \subseteq \mathcal{X} \times \mathcal{X}'$ denotes the defense relation, where $(w, w') \in \mathcal{R}$ indicates that $w'$ is a valid defended trace generated from the original trace $w$ by the defense mechanism. A WF defense mechanism usually modifies $w$ to $w'$ by injecting dummy packets $d_i'$, where $i$ denotes the dummy packet's insertion position in the trace.

In this work, we propose $\mathcal{D}$, a deep learning-based approach that learns defense-induced traffic transformation patterns from paired original and defended Tor traffic traces. Given a training dataset:

$$
\mathcal{S}_{\mathrm{train}} = \{(w_i, w_i')\}_{i=1}^{M}
$$

where each $w_i$ is an original trace and $w_i'$ is its corresponding defended trace, the model learns to identify the injected dummy packets and recover the original packet-direction sequence.

After training, the model is applied to previously unseen defended traces to remove injected dummy packets and reconstruct the underlying original traffic. Formally, the recovery process is defined as:

$$
\hat{w} = \mathcal{G}(w')
$$

where $\mathcal{G}$ denotes the trained recovery model, and $\hat{w}$ is the recovered trace approximating the original trace $w$. The objective is to minimize the discrepancy between the recovered trace $\hat{w}$ and its original counterpart $w$, enabling the model to recover original traffic patterns from defended observations.

Since $w$ and $w'$ are sequences of packet directions, and the defense mechanism preserves the original packets while inserting dummy packets, the original trace $w$ is a subsequence of the defended trace $w'$, denoted by:

$$
w \preceq w'
$$

Accordingly, the recovery model aims to identify and remove the injected dummy packets from $w'$ to reconstruct the original trace $w$.

### Notation and Definitions

![Table 1](./images/table1.png)

Table 1 summarizes the key symbols and terminology used throughout this work. $B$ denotes the mini-batch size used during model training, which is set to 128 by default. $L$ represents the padded sequence length, with a default value of 10,000, ensuring that traffic traces can be processed in a consistent input format. $K$ denotes the number of support traces per class in the few-shot learning setting, which is set to 5 by default. The variable $\ell$ represents the actual length of a defended traffic trace before padding, satisfying $\ell \leq L$. Finally, $LCS$ stands for Longest Common Subsequence, a measure used to identify the longest sequence of elements shared by two sequences while preserving their relative order.

## System Design

![Architecture](./images/daae.png)
Figure 2: An overview of the DAAE architecture.

In this section, we demonstrate the design details of the DAAE. As shown in Figure 2, the proposed DAAE comprises three key phases: the dataset pre-processing phase, the few-shot-based DAAE (FS-DAAE) model training phase, and the defended trace recovery phase.

### Dataset Pre-processing

In this work, the proposed $\mathcal{D}$ aims to learn the dummy packet injection patterns of existing WF defense mechanisms by training a deep learning model on a small number of paired original and defended Tor traces. Considering that some defense mechanisms can double or even triple the length of a trace, we set the trace length to $L=10000$ to accommodate the additional packets introduced by these defenses.

Let $\mathbf{x}^{\mathrm{ori}}=[d^{\mathrm{ori}}_1,\ldots,d^{\mathrm{ori}}_n]$ and $\mathbf{x}^{\mathrm{def}}=[d^{\mathrm{def}}_1,\ldots,d^{\mathrm{def}}_m]$ denote the direction sequences of the original and defended traces, respectively, where each direction is represented by an element of $\{\pm 1\}$ and $m \geq n$ in the typical case. Since the insertion of dummy packets shifts the positions of subsequent original packets, the two traces can no longer be aligned directly by packet index.

To address this issue, DAAE formulates dummy packet identification as an **insertion-only sequence alignment problem** and employs the longest common subsequence (LCS) algorithm to align the two direction sequences. The LCS is computed using the following dynamic programming recurrence:

$$
\mathrm{LCS}[i,j] =
\begin{cases}
0, & \text{if } i=0 \text{ or } j=0,\\
\mathrm{LCS}[i-1,j-1]+1,
& \text{if } d^{\mathrm{ori}}_i = d^{\mathrm{def}}_j,\\
\max\{\mathrm{LCS}[i-1,j],\,\mathrm{LCS}[i,j-1]\},
& \text{otherwise}.
\end{cases}
$$

By applying LCS-based alignment to each pair of original and defended traces, we can identify the positions of injected dummy packets in the defended trace. Based on the resulting alignment, we construct an **LCS-based keep mask** to label each packet in the defended trace: a value of 1 indicates that the packet corresponds to an original packet, whereas a value of 0 indicates that it is a dummy packet. The resulting keep masks serve as supervision labels for training the FS-DAAE model, enabling it to learn dummy packet injection patterns from a limited number of paired traces.

The support LCS bank maintains a persistent, class-conditioned memory of aligned defended traces and their LCS-derived keep masks, enabling defended trace recovery without requiring the original capture of a query trace. For each website class, it stores up to $K$ training instances and provides two complementary views: a few-shot support set for model training and a class-specific prototype for decoding. The prototype is constructed from the longest retained original-packet subsequence among the support instances, while the maximum keep ratio across these instances serves as a class-specific target for calibrating insertion-only Viterbi decoding. Unlike a conventional classifier memory that stores learned embeddings, the support LCS bank preserves representative defended traffic patterns and their packet-level original/dummy labels. The detailed FS-DAAE model training process will be introduced in the next section.

### FS-DAAE Model Training

The proposed DAAE model is trained in a supervised few-shot setting (FS-DAAE) using paired original and defended Tor traces obtained during dataset preprocessing. Each training episode comprises a query trace and $K=5$ support traces sampled from the same website class, with the query excluded from support selection. Each pair is aligned using the LCS algorithm, providing packet-level supervision for distinguishing original packets from injected dummy packets.  This episodic formulation enables FS-DAAE to learn specific defense patterns across different visits without requiring the original trace of a query at inference time (defended trace recovery).

*FS-DAAE architecture.* FS-DAAE employs a one-dimensional U-Net and a four-stage stride-2 encoder. The four-stage stride-2 encoder progressively reduces the input sequence length from $L=10000$ to $L/16=625$, while increasing the feature channels from 1 to 512. The query and support traces share encoder weights. At the bottleneck, the LCS-derived support masks are embedded into the support features, which are aggregated to form a class-conditioned representation. Feature-wise linear modulation and eight-head cross-attention then incorporate support information into the query representation. A skip-connected decoder restores the sequence to its original resolution, incorporating an averaged support keep prior at the finest scale. A four-branch dilated mask head with dilation rates $\{1,2,4,8\}$ captures packet-retention patterns at multiple receptive-field scales. An auxiliary reconstruction head predicts packet directions, while a pooled latent representation is learned implicitly through the network. The primary output is a sequence of mask logits $\mathbf{z}\in\mathbb{R}^{B\times L}$, where $B$ is the batch size, and the reconstruction head produces $\hat{\mathbf{x}}\in\mathbb{R}^{B\times L}$.

*FS-DAAE Training Objective.* FS-DAAE is supervised by two signals derived from LCS alignment: a keep mask $\mathbf{M}$, where 1 indicates an original packet and 0 indicates a dummy packet, and a reconstruction target $\mathbf{x}^{\mathrm{target}}$ defined on the defended-trace timeline. At each matched position, the target contains the corresponding original packet direction. The unmatched dummy positions are assigned zero and excluded from reconstruction loss. A validity mask $V_{b,j}=\mathbb{1}[j<\ell_b]$, where $\ell_b$ is the valid length of query $b$, excludes padded positions from all loss terms.

The overall FS-DAAE training objective combines three loss groups comprising four complementary components: ℒ<sub>mask</sub>, ℒ<sub>recon</sub>, and ℒ<sub>len</sub>. The total objective is:

$$
ℒ = \lambda_{\mathrm{mask}}ℒ_{\mathrm{mask}} + ℒ_{\mathrm{recon}} + ℒ_{\mathrm{len}}
$$

where the default weight is $\lambda_{\mathrm{mask}}=2$. Specifically,

$$
ℒ_{\mathrm{mask}} = ℒ_{\mathrm{focal}} + \lambda_{\mathrm{dice}}ℒ_{\mathrm{dice}}
$$

where the default $\lambda_{\mathrm{dice}}$ is 0.5.

*Reconstruction loss ℒ<sub>recon</sub>:* Mean squared error is evaluated only on LCS-kept, unpadded defended positions. Dummy and pad indices are excluded from both the numerator and the denominator:

$$
ℒ_{\mathrm{recon}} =
\frac{
\sum_{b=1}^{B}\sum_{j=0}^{L-1}
V_{bj}M_{bj}
\left(\hat{x}_{bj}-x^{\mathrm{target}}_{bj}\right)^2
}{
\sum_{b=1}^{B}\sum_{j=0}^{L-1}
V_{bj}M_{bj}
}
$$

*Length loss ℒ<sub>len</sub>:* The predicted keep fraction is the mean of the *sigmoid probabilities* $p_{bj}$ over valid positions of each trace, not a count of thresholded decisions $\mathbb{1}[\sigma(z_{bj})\geq 0.5]$. With the time changed, the trace length will increase, so that:

$$
p^{\mathrm{t}}_{bj} = p_{bj} M_{bj} + (1-p_{bj})(1-M_{bj})
$$

For each $b$:

$$
\hat{r}_b =
\frac{\sum_{j=0}^{L-1} p_{bj} V_{bj}}
{\sum_{j=0}^{L-1} V_{bj}},
\qquad
r'_b =
\frac{\sum_{j=0}^{L-1} M_{bj} V_{bj}}
{\sum_{j=0}^{L-1} V_{bj}}
$$

The length term is the batch-mean $\ell_1$ gap between these soft rates:

$$
\mathcal{L}_{\mathrm{len}} =
\frac{1}{B}
\sum_{b=1}^{B}
\left|\hat{r}_b-r'_b\right|
$$

### Defended Trace Recovery

The recovery phase reconstructs original traffic traces from defended-only observations, without requiring access to the query's original trace during inference. It uses the FS-DAAE checkpoint selected by the highest validation mask accuracy and the persistent support LCS bank constructed from the training split. For each defended capture, packet directions are extracted and truncated to a maximum sequence length $L=10000$, with the valid trace length recorded separately. All support information comes exclusively from training data, ensuring that recovery does not depend on the query's original trace or validation/test samples.

#### FS-DAAE inference. 
The defended trace query will be passed through FS-DAAE in evaluation mode without gradient computation. The network applies the same support–query fusion used during training, including mask-conditioned support encoding, feature-wise linear modulation, cross-attention, and the U-Net decoder, to produce packet-level keep logits $\mathbf{z}\in\mathbb{R}^{ℒ}$. Although thresholding these logits at zero yields a straightforward binary keep mask, this strategy can retain isolated dummy packets or discard original packets. Therefore, the default recovery procedure uses insertion-only Viterbi decoding to exploit the structural constraint that the original direction sequence is a subsequence of the defended sequence.

#### Trace Reconstruction.
Finally, DAAE retains the directions of all defended packets selected by the decoded keep mask, discarding the remaining packets and preserving their original order. Recovery is performed independently of evaluation: original traces are used only for subsequent assessment, not for generating the recovered output. In particular, the reported recovery results use the support-bank and insertion-decoding pipeline without oracle access to the query original. This design enables DAAE to remove dummy packets by combining learned packet-level evidence, specific defense patterns, and insertion-constrained sequence alignment.

## Performance Evaluation

### DAAE Decoding Strategy

![Figure 3](./images/figure3.png)

At test time, DAAE does not have access to the original trace. Instead, it maintains a small support pool constructed from the training set and applies the same inference pipeline to the captured defended traffic.
We designed two strategies to decode the defended packets into the original ones.
We find that directly thresholding at 0.5 for the predicted keep probabilities can incorrectly remove original packets or retain dummy packets, disrupting the original packet-direction subsequence. Therefore, we perform a sequence-level decoding step against support prototypes, where each prototype represents the packet-direction sequence of the packets labeled as original in a support trace. We introduce a skip bias to control the trade-off between retaining and removing packets and select the bias that produces a keep rate closest to the median keep rate of the support set. When multiple supports are available, we first consider supports whose kept-trace lengths are closest to the target length and select the candidate yielding the best sequence-level decoding score. The recovered trace is then constructed by retaining the captured packets selected by the final mask. Finally, we train a standard WF classifier, such as DF and RF, on the recovered traces and evaluate its classification performance to quantify how effectively DAAE restores the fingerprint information removed by the defense.

As shown in the third column of Figure 3, the recovered traces do not exactly match the originals, but much of their original traffic structure is restored. This result suggests that defense-induced traffic artifacts can be exploited to partially reverse the effect of traffic regularization, potentially undermining the defense against a sufficiently adaptive adversary. Figure 3 demonstrates that the original non-defended traces show distinct patterns in their Traffic Aggregation Matrices (TAMs). Palette aims to reduce this variability by regularizing traffic into a common pattern. However, the resulting defended traces can still preserve noticeable differences. More importantly, the regularization process introduces systematic dummy-packet injection patterns that may themselves become learnable by an adaptive adversary. 


### Closed-world Evaluation

![Table 2](./images/table2.png)

Table 2 presents a comprehensive comparison of the proposed \D-based WF attacks against conventional WF attacks trained with and without adversarial training (w/ AT and w/o AT, respectively). Without adversarial training (w/o AT), most defenses substantially reduce the classification accuracy of all three attacks, particularly on the DF dataset, where the accuracy of DF, Var-CNN, and NetCLR drops to 0.97–27.54% for most defenses. However, this apparent protection weakens considerably when the attacks are trained with adversarial training (w/ AT). For example, Walkie-Talkie, FRONT, and FRUGAL exhibit accuracies above 95% against all three attacks on the DF dataset, while GAPDiS also yields high accuracies ranging from 88.23% to 95.17%. These results suggest that evaluating defenses solely against conventionally trained attacks can substantially overestimate their robustness. The DAAE-based attacks further expose this limitation: their classification accuracy reaches 97.85% against FRUGAL and exceeds 95% against RegulaTor for all three classifiers on the DF dataset. Although some defenses, such as Palette and Adaptive Tamaraw, exhibit lower accuracy under adversarial training, their effectiveness remains dependent on the attack model, highlighting the importance of evaluating defenses under diverse attacker capabilities.
The results on GTT23 further reveal that defense effectiveness does not necessarily transfer across datasets. Compared with DF, the accuracy of conventionally trained attacks is generally higher against several defenses on GTT23; for example, FRONT yields accuracies of 35.94% and 52.30% against DF and NetCLR, respectively, while its corresponding accuracies on DF are 27.54% and 25.79%. Adversarial training also substantially improves attack performance against most defenses on GTT23, with accuracies reaching 89.24% for NetCLR against GAPDiS and 86.72% for NetCLR against FRONT. The DAAE-based attacks maintain high accuracy against several defenses, including GAPDiS, FRUGAL, and RegulaTor, although the results vary across classifiers. Notably, RegulaTor and Palette exhibit relatively low accuracy under adversarial training on both datasets, but their DAAE-based results show that this apparent robustness can be weakened by stronger attack strategies. Overall, these findings indicate that WF defense evaluations should account for both adversarially trained attackers and \D-based attacks, as well as differences between datasets, to provide a more comprehensive assessment of defense robustness.

### Open-world Evaluation

![Table3](./images/table3.png)
The results in Table 3 show that the effectiveness of WF defenses varies considerably across attack models and training settings. Without adversarial training, most defenses exhibit limited resistance to WF attacks, although their performance differs across classifiers. For example, FRONT achieves relatively high TPRs of 68.72% and 66.74% against DF and Var-CNN, respectively, while maintaining moderate FPRs. In contrast, several defenses, including RegulaTor, Palette, Adaptive Tamaraw, and FRUGAL, exhibit substantially lower TPRs or high FPRs, indicating that their protection effectiveness is inconsistent. Adversarial training considerably improves attack performance across nearly all defenses, with TPRs frequently exceeding 90%. This suggests that adversarial training strengthens the robustness of WF classifiers against traffic perturbations introduced by these defenses. However, the results under DAAE-based attacks reveal that high detection rates largely persist even when defenses are evaluated under this setting: most defenses achieve TPRs above 82% across all three classifiers, with FPRs generally remaining below 16%. Notably, GAPDiS and FRONT achieve consistently high TPRs across the three classifiers, while Walkie-Talkie also exhibits strong detection performance, particularly under DF and Var-CNN. Overall, these findings indicate that existing SOTA defenses provide limited protection against advanced WF attacks in the open-world setting, and that adversarial training alone does not eliminate the vulnerability.
