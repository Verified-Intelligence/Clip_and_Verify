# Clip-and-Verify: Linear Constraint-Driven Domain Clipping for Accelerating Neural Network Verification

State-of-the-art neural network verifiers like α,β-CROWN combine branch-and-bound (BaB) with fast bound propagation to tackle challenging verification problems at scale. However, existing implementations typically compute intermediate bounds once at initialization and then only optimize the final layer. This design keeps things scalable, but wastes many opportunities to tighten the relaxation deeper in the network, especially on hard BaB subproblems.  

**Clip-and-Verify** introduces a *linear constraint-driven clipping* framework that exploits the linear constraints generated “for free” during bound propagation and BaB (e.g., output constraints, activation split constraints) to aggressively shrink the effective input box and tighten intermediate bounds at any layer. Under this framework, we develop two specialized GPU algorithms:

* **Relaxed Clipping**: cheaply shrinks the input domain via closed-form 1D dual updates and then re-concretizes cached linear bounds over the tightened box.
* **Complete Clipping**: runs a coordinate-ascent procedure over Lagrange multipliers to directly refine intermediate bounds, achieving near-LP tightness at a fraction of LP’s cost.  

We integrate these clipping routines into both **input-space BaB** and **activation-space BaB** in α,β-CROWN, using either the accumulated split constraints or output constraints to progressively prune infeasible regions and reduce the number of hard subproblems. Across benchmarks from **VNN-COMP 2021–2024** and control systems, Clip-and-Verify:

* cuts BaB branches by **>50%** and up to **96% fewer subproblems** on hard control tasks,
* consistently improves verified accuracy and runtime when combined with β-CROWN or BICCOS,
* attains **state-of-the-art verification coverage** on challenging VNN-COMP benchmarks such as *cifar10-resnet*, *cifar100-2024*, *tinyimagenet-2024*, and *vit-2024*, especially when combined with BICCOS.  

Clip-and-Verify is part of the **α,β-CROWN** verifier, which is the **VNN-COMP 2025 overall winner**. 

More details can be found in our paper: 

[**Clip-and-Verify: Linear Constraint-Driven Domain Clipping for Accelerating Neural Network Verification**](https://openreview.net/pdf?id=HuSSR12Yot) 
**NeurIPS 2025** (39th Conference on Neural Information Processing Systems) 
Duo Zhou*, Jorge Chavez*, Hesun Chen, Grani A. Hanasusanto, Huan Zhang (*equal contribution) 

Code: [https://github.com/Verified-Intelligence/alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN) 

<p align="center">
<a href="https://github.com/Verified-Intelligence/alpha-beta-CROWN">
<img src="https://www.huan-zhang.com/images/upload/alpha-beta-crown/logo_2022.png" width="28%">
</a>
</p>

## Reproducing results

Clip-and-Verify is implemented on top of our **α,β-CROWN** verifier (alpha-beta-CROWN), which is the winning tool of **VNN-COMP 2021, 2022, 2023, 2024, and 2025**. All experiments in the paper are run inside α,β-CROWN using configuration files that enable the Clip-and-Verify pipeline. ([α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN))

This repository provides ready-to-use configuration files and scripts for the experiments in the paper. Below we outline the typical setup and command patterns; please adjust paths and config filenames to match the actual files in this repo and in α,β-CROWN.

### 1. Install α,β-CROWN

First, clone and install the α,β-CROWN verifier (which includes the `auto_LiRPA` submodule):

```bash
# Clone α,β-CROWN with its submodule
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
cd alpha-beta-CROWN

# (Optional) remove an old environment with the same name
conda deactivate || true
conda env remove --name alpha-beta-crown || true

# Create and activate the environment (Python 3.11, PyTorch 2.3.1)
conda env create -f complete_verifier/environment.yaml --name alpha-beta-crown
conda activate alpha-beta-crown
```

Alternatively, you can follow the `pip`-based installation described in the α,β-CROWN README if you want to use an existing environment. ([GitHub][1])

> **Note:** Clip-and-Verify itself does **not** require any external LP/MIP solvers; all clipping routines run on GPU without Gurobi/CPLEX. You only need such solvers if you explicitly enable other MIP-based algorithms in α,β-CROWN. 

### 2. Download Benckmarks

All benckmarks are opensouced VNN-COMP benckmarks. Get the benckmarks from the VNN-COMP official github:

```bash
git clone https://github.com/stanleybak/vnncomp2021.git

git clone https://github.com/ChristopherBrix/vnncomp2022_benchmarks.git
# cd vnncomp2022_benchmarks
# bash setup.sh

git clone https://github.com/ChristopherBrix/vnncomp2023_benchmarks.git

git clone https://github.com/ChristopherBrix/vnncomp2024_benchmarks.git
```

To reproduce the results in control verification problems, please also clone this repository:

```bash
git clone https://github.com/Verified-Intelligence/Clip_and_Verify.git
```


### 3. Running Clip-and-Verify experiments

All experiments use the unified front-end `abcrown.py` from α,β-CROWN; Clip-and-Verify is enabled through the YAML config options (e.g., toggling relaxed vs. complete clipping, number of clipping rounds, which constraints are used, etc.). ([GitHub][1])

Below are typical command patterns for different benchmark groups. Replace `<config>` with the actual config filenames provided in this repo.

#### 3.1 Input-space BaB benchmarksfrom VNN-COMP

For input-space BaB benchmarks such as **acasxu** (VNN-COMP 2021), **lsnc** (VNN-COMP 2024), **nn4sys** (VNN-COMP 2022), we use Relaxed and/or Complete Clipping to tighten the input domain and prune subproblems. Listed in this subsection are the command lines to run the VNN-COMP benchmarks.

* Clip-n-Verify, Complete

    ```
    # acasxu
    python abcrown.py --config exp_configs/vnncomp23/acasxu.yaml --enable_clip_input --clip_input_type complete --reorder_bab

    # nn4sys
    python abcrown.py --config exp_configs/vnncomp23/nn4sys.yaml --enable_clip_input --clip_input_type complete --reorder_bab

    # lsnc
    python abcrown.py --config exp_configs/vnncomp24/lsnc.yaml --enable_clip_input --clip_input_type complete --reorder_bab
    ```

* Clip-n-Verify, Relaxed

    ```
    python abcrown.py --config exp_configs/vnncomp23/acasxu.yaml --enable_clip_input --clip_input_type relaxed --reorder_bab

    python abcrown.py --config exp_configs/vnncomp23/nn4sys.yaml --enable_clip_input --clip_input_type relaxed --reorder_bab 

    python abcrown.py --config exp_configs/vnncomp24/lsnc.yaml --enable_clip_input --clip_input_type relaxed --reorder_bab
    ```

* Clip-n-Verify, Relaxed w/o Reorder

    ```
    python abcrown.py --config exp_configs/vnncomp23/acasxu.yaml --enable_clip_input --clip_input_type relaxed

    python abcrown.py --config exp_configs/vnncomp23/nn4sys.yaml --enable_clip_input --clip_input_type relaxed

    python abcrown.py --config exp_configs/vnncomp24/lsnc.yaml --enable_clip_input --clip_input_type relaxed
    ```


These configs reproduce the results in Table 1 of the paper (branch reductions and subproblem reductions up to 96% on the hardest control tasks). 

#### 3.2 Hard Control Verification Problem

The NN control system verificaion problems in our paper come from a recent study on [**provably stable neural network control systems**](https://github.com/Verified-Intelligence/Two-Stage_Neural_Controller_Training). In Table 2, we illustrated the results on 3 different problems. To reproduce the results, please run the following command lines:

* Clip-and-Verify, complete

    ```bash
    python abcrown.py   --config ../../Clip_and_Verify/Control_Verification/cartpole/cartpole.yaml --enable_clip_input --clip_input_type complete

    python abcrown.py   --config ../../Clip_and_Verify/Control_Verification/quadrotor/quadrotor.yaml --enable_clip_input --clip_input_type complete

    python abcrown.py   --config ../../Clip_and_Verify/Control_Verification/quadrotor_large/quadrotor_large.yaml --enable_clip_input --clip_input_type complete
    ```

*   Clip-and-Verify, relaxed

     ```bash
    python abcrown.py   --config ../../Clip_and_Verify/Control_Verification/cartpole/cartpole.yaml --enable_clip_input

    python abcrown.py   --config ../../Clip_and_Verify/Control_Verification/quadrotor/quadrotor.yaml --enable_clip_input

    python abcrown.py   --config ../../Clip_and_Verify/Control_Verification/quadrotor_large/quadrotor_large.yaml --enable_clip_input
    ```

*   Clip-and-Verify, $\alpha,\beta$-CROWN
    
    ```bash
    python abcrown.py   --config ../../Clip_and_Verify/Control_Verification/cartpole/cartpole.yaml

    python abcrown.py   --config ../../Clip_and_Verify/Control_Verification/quadrotor/quadrotor.yaml

    python abcrown.py   --config ../../Clip_and_Verify/Control_Verification/quadrotor_large/quadrotor_large.yaml
    ```


#### 3.3 Activation-space BaB benchmarks from VNN-COMP

For activation-space BaB benchmarks from **VNN-COMP 2021–2024** (e.g., *oval22*, *cifar10-resnet*, *cifar100-2024*, *tinyimagenet-2024*, *vit-2024*), we integrate Complete Clipping into the β-CROWN / BICCOS BaB loop. 

Example:

* Run Clip and Verify with β-CROWN

    Remove `--enable_complete_clip` to run vanilla β-CROWN.

    ```
    python abcrown.py --config exp_configs/beta_crown/cifar_cnn_a_mix.yaml  --enable_complete_clip  --clip_alpha_crown 

    python abcrown.py --config exp_configs/beta_crown/cifar_cnn_a_mix4.yaml  --enable_complete_clip  --clip_alpha_crown 

    python abcrown.py --config exp_configs/beta_crown/cifar_cnn_a_adv.yaml  --enable_complete_clip  --clip_alpha_crown 

    python abcrown.py --config exp_configs/beta_crown/cifar_cnn_a_adv4.yaml  --enable_complete_clip  --clip_alpha_crown 

    python abcrown.py --config exp_configs/beta_crown/cifar_cnn_b_adv.yaml  --enable_complete_clip  --clip_alpha_crown 

    python abcrown.py --config exp_configs/beta_crown/cifar_cnn_b_adv4.yaml  --enable_complete_clip  --clip_alpha_crown 

    python abcrown.py --config exp_configs/beta_crown/mnist_cnn_a_adv.yaml  --enable_complete_clip --clip_alpha_crown 

    python abcrown.py --config exp_configs/vnncomp21/cifar10-resnet.yaml  --enable_complete_clip  --clip_alpha_crown 

    python abcrown.py --config exp_configs/vnncomp22/oval22.yaml  --enable_complete_clip  --clip_alpha_crown 

    python abcrown.py --config exp_configs/vnncomp24/cifar100.yaml  --enable_complete_clip --clip_alpha_crown 

    python abcrown.py --config exp_configs/vnncomp24/tinyimagenet.yaml  --enable_complete_clip --clip_alpha_crown 

    python abcrown.py --config exp_configs/vnncomp23/vit.yaml  --enable_complete_clip --clip_alpha_crown 
    ```

* Run Clip and Verify with GCP-CROWN with mip cuts

    First, cd `./cuts/CPLEX_cuts` and follow the instrction by `README.md` to complie cut solver.

    Remove `--enable_complete_clip` to run vanilla GCP-CROWN.

    ```
    python abcrown.py --config exp_configs/GCP-CROWN/cifar_cnn_a_mix.yaml  --enable_complete_clip --clip_alpha_crown 

    python abcrown.py --config exp_configs/GCP-CROWN/cifar_cnn_a_mix4.yaml  --enable_complete_clip --clip_alpha_crown 
 
    python abcrown.py --config exp_configs/GCP-CROWN/cifar_cnn_a_adv.yaml  --enable_complete_clip --clip_alpha_crown 

    python abcrown.py --config exp_configs/GCP-CROWN/cifar_cnn_a_adv4.yaml  --enable_complete_clip --clip_alpha_crown 


    python abcrown.py --config exp_configs/GCP-CROWN/cifar_cnn_b_adv.yaml  --enable_complete_clip --clip_alpha_crown 


    python abcrown.py --config exp_configs/GCP-CROWN/cifar_cnn_b_adv4.yaml  --enable_complete_clip --clip_alpha_crown 


    python abcrown.py --config exp_configs/GCP-CROWN/mnist_cnn_a_adv.yaml  --enable_complete_clip --clip_alpha_crown 


    python abcrown.py --config exp_configs/GCP-CROWN/cifar10-resnet.yaml  --enable_complete_clip --clip_alpha_crown 


    python abcrown.py --config exp_configs/vnncomp22/oval22.yaml  --enable_complete_clip --clip_alpha_crown 

    ```

* Run Clip and Verify with GCP-CROWN with BICCOS

    Remove `--enable_complete_clip` to run vanilla BICCOS.

    ```
    python abcrown.py --config exp_configs/BICCOS/cifar_cnn_a_mix/biccos_all_selective_mts_plus_gcp.yaml --enable_complete_clip --clip_alpha_crown 


    python abcrown.py --config exp_configs/BICCOS/cifar_cnn_a_mix4/biccos_all_selective_mts_plus_gcp.yaml --enable_complete_clip   --clip_alpha_crown 


    python abcrown.py --config exp_configs/BICCOS/cifar_cnn_a_adv/biccos_all_selective_mts_plus_gcp.yaml --enable_complete_clip   --clip_alpha_crown 


    python abcrown.py --config exp_configs/BICCOS/cifar_cnn_a_adv4/biccos_all_selective_mts_plus_gcp.yaml --enable_complete_clip   --clip_alpha_crown 


    python abcrown.py --config exp_configs/BICCOS/cifar_cnn_b_adv/biccos_all_selective_mts_plus_gcp.yaml --enable_complete_clip  --clip_alpha_crown 


    python abcrown.py --config exp_configs/BICCOS/cifar_cnn_b_adv4/biccos_all_selective_mts_plus_gcp.yaml --enable_complete_clip   --clip_alpha_crown 


    python abcrown.py --config exp_configs/BICCOS/mnist_cnn_a_adv/biccos_all_selective_mts_plus_gcp.yaml --enable_complete_clip   --clip_alpha_crown 


    python abcrown.py --config exp_configs/BICCOS/oval22/biccos_all_selective_mts_plus_gcp.yaml --enable_complete_clip  --clip_alpha_crown 
 

    python abcrown.py --config exp_configs/BICCOS/cifar10_resnet/biccos_all_selective_mts_plus_gcp.yaml --enable_complete_clip   --clip_alpha_crown 


    python abcrown.py --config exp_configs/BICCOS/cifar100/biccos_all_selective_mts.yaml --enable_complete_clip   --clip_alpha_crown 


    python abcrown.py --config exp_configs/BICCOS/tinyimagenet/biccos_all_selective_mts.yaml --enable_complete_clip --clip_alpha_crown 

    ```


These runs reproduce the verified-accuracy and runtime comparisons in Tables 3–4, where **Clip-and-Verify with BICCOS** attains state-of-the-art coverage and pushes closer to the theoretical upper bounds. 

## BibTeX Entry

If you use Clip-and-Verify in your work, please cite our paper:

```bibtex
@inproceedings{
    zhou2025clipandverify,
    title={Clip-and-Verify: Linear Constraint-Driven Domain Clipping for Accelerating Neural Network Verification},
    author={Duo Zhou and Jorge Chavez and Hesun Chen and Grani A. Hanasusanto and Huan Zhang},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
    url={https://openreview.net/forum?id=HuSSR12Yot}
}
```

For citing the underlying verifier and related components (CROWN, α-CROWN, β-CROWN, BICCOS, etc.), please also see the citation recommendations in the α,β-CROWN repository [1].

[1]: https://github.com/Verified-Intelligence/alpha-beta-CROWN "GitHub - Verified-Intelligence/alpha-beta-CROWN: alpha-beta-CROWN: An Efficient, Scalable and GPU Accelerated Neural Network Verifier (winner of VNN-COMP 2021, 2022, 2023, 2024, 2025)"
