# This are the models succesfully trined and setting used

## RNN based

### tahn

- Current version

  ```yml
  training:
  num_epochs: 5000
  batch_size: 2048
  learning_rate: 0.001
  n_trials: 1                # independent training runs (each saved separately)
  early_stopping_patience: 20
  early_stopping_min_delta: 1.0e-5  # minimum improvement to reset patience counter
  loss_func: 'MSE'           # 'MSE' | 'Huber'
  split_train: 70            # % of dataset used for training
  split_val: 20              # % for validation  (remainder → test, not used in loop)
  split_test: 10
  max_rollout_steps: 5       # replaces hardcoded 5; val also uses this
  rollout_gamma: 0.95        # replaces hardcoded 0.9
  stateful_rollout: true     # thread hidden state through rollout (RNN/LSTMNN only)
  lr_phase_decay: 0.75       # multiply LR by this factor at each phase advance
  lr_scheduler_patience: 10  # ReduceLROnPlateau patience within a phase
  lr_scheduler_factor: 0.5   # ReduceLROnPlateau reduction factor
  ```

## LSTMNN

The best model so far is `LSTMNN_L63_trial1_1778503362` but there are still many thing to improve