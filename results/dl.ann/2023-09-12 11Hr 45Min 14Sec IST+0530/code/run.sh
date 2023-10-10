if [ "$1" = "ann" ]; then
    python train.py \
        model=deep_learning/ann \
        model.hidden_layers=[10,10,10] \
        model.use_batch_norm=True \
        model.use_lora=False \
        trainer.optim=adam \
        trainer.batch_size=256 \
        trainer.epochs=100 \
        trainer.policy_exp_path=False \
        data.patch.stride=150 \
        data.patch.patch_size=150 \
        data.class_weights=[1,1,1,1,1,1] \
        data.drop_columns=['W_Tar','lat','lng','DEPT','RW'] \
        data.scaled_columns=['GR','NPHI','DPHI','ILD','VSH'] \
        callbacks.early_stopping_tolerance=10 \
        comment="undefined class removed and filled with prediction of random formest" \
        2>&1 | tee output.log

elif [ "$1" = "vit" ]; then
    if [ "$2" = "lora" ]; then
        if [ "$3" = "without" ]; then
            python train.py \
                model=deep_learning/vit \
                model.activation=relu \
                model.autoregressive=False \
                model.dim=150 \
                model.dim_head=256 \
                model.heads=4 \
                model.depth=1 \
                model.mlp_dim=128 \
                model.auto_regressor_hidden_layer_sizes=[150] \
                model.use_lora=False \
                trainer.optim=adam \
                trainer.batch_size=64 \
                trainer.epochs=200 \
                trainer.policy_exp_path=False \
                data.patch.stride=150 \
                data.patch.patch_size=150 \
                data.class_weights=[1,1,1,1,1,1] \
                data.drop_columns=['W_Tar','lat','lng','DEPT','RW'] \
                data.scaled_columns=['GR','NPHI','DPHI','ILD','VSH'] \
                data.x_file_name="lora_X.h5" \
                data.y_file_name="lora_Y.h5" \
                callbacks.early_stopping_tolerance=10 \
                comment="used for LoRA comparison study-training from scratch" \
                2>&1 | tee output.log
        elif [ "$3" = "tl" ]; then
            python train.py \
                model=deep_learning/vit \
                model.activation=relu \
                model.autoregressive=False \
                model.dim=150 \
                model.dim_head=256 \
                model.heads=4 \
                model.depth=1 \
                model.mlp_dim=128 \
                model.auto_regressor_hidden_layer_sizes=[150] \
                model.use_lora=False \
                trainer.optim=adam \
                trainer.batch_size=64 \
                trainer.epochs=200 \
                trainer.policy_exp_path='/home/nasim/phd/petro_AViT/results/dl.vit/2023-09-11 11Hr 50Min 06Sec IST+0530' \
                data.patch.stride=150 \
                data.patch.patch_size=150 \
                data.class_weights=[1,1,1,1,1,1] \
                data.drop_columns=['W_Tar','lat','lng','DEPT','RW'] \
                data.scaled_columns=['GR','NPHI','DPHI','ILD','VSH'] \
                data.x_file_name="lora_X.h5" \
                data.y_file_name="lora_Y.h5" \
                callbacks.early_stopping_tolerance=10 \
                comment="used for LoRA comparison study-training from already trained model" \
                2>&1 | tee output.log
        else
            python train.py \
                model=deep_learning/vit \
                model.activation=relu \
                model.autoregressive=False \
                model.dim=150 \
                model.dim_head=256 \
                model.heads=4 \
                model.depth=1 \
                model.mlp_dim=128 \
                model.auto_regressor_hidden_layer_sizes=[150] \
                model.use_lora=True \
                trainer.optim=adam \
                trainer.batch_size=64 \
                trainer.epochs=200 \
                data.patch.stride=150 \
                data.patch.patch_size=150 \
                data.class_weights=[1,1,1,1,1,1] \
                data.drop_columns=['W_Tar','lat','lng','DEPT','RW'] \
                data.scaled_columns=['GR','NPHI','DPHI','ILD','VSH'] \
                callbacks.early_stopping_tolerance=10 \
                comment="used for LoRA comparison study-training from LoRA model with already trained model" \
                2>&1 | tee output.log
        fi
    else 
        python train.py \
            model=deep_learning/vit \
            model.activation=relu \
            model.autoregressive=False \
            model.dim=150 \
            model.dim_head=256 \
            model.heads=4 \
            model.depth=1 \
            model.mlp_dim=128 \
            model.auto_regressor_hidden_layer_sizes=[8,32,64,150] \
            model.use_lora=False \
            trainer.optim=adam \
            trainer.batch_size=64 \
            trainer.epochs=200 \
            data.patch.stride=150 \
            data.patch.patch_size=150 \
            data.class_weights=[1,1,1,1,1,1] \
            data.drop_columns=['W_Tar','lat','lng','DEPT','RW'] \
            data.scaled_columns=['GR','NPHI','DPHI','ILD','VSH'] \
            callbacks.early_stopping_tolerance=10 \
            comment="undefined class removed and filled with prediction of random formest" \
            2>&1 | tee output.log
    fi

else
    echo "Model Type can be either ann or vit"
fi
