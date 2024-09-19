from transformers import Trainer, AutoModelForCausalLM


def model_init(model_name, model_type, config):
    if model_type == 'GPT':
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            from_tf=bool(".ckpt" in model_name),
            config=config,
            )

def hyper_param_search(training_args, hp_space, train_set, val_set, data_collator):
    
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=data_collator,
    )

    best_trial = trainer.hyperparameter_search(
        direction="minimize",
        backend="optuna",
        hp_space=hp_space,
        n_trials=10,
    )

    return best_trial.hyperparameters