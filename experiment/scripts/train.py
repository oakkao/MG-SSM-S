import torch
import torch.nn as nn
import torch.optim as optim

# Model training part 
def ModelTraining(model_config , train_data, eval_data, test_data, num_epochs, learning_rate, num_runs, loss_function = nn.MSELoss(), eval_function = nn.MSELoss()):  #train_raw_loader, test_raw_loader
  # Initialize a list to store test losses
  test_losses = []
  model_function, config = model_config
  model_name = model_function.__name__
  for run in range(num_runs):
      print(f"\n--- {model_name} Run {run + 1}/{num_runs} ---")

      # Initialize the model
      model = model_function(**config).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

      # Define loss function and optimizer
      criterion = loss_function
      optimizer = optim.Adam(model.parameters(), lr=learning_rate)

      # Early stopping parameters
      best_eval_loss = float('inf')
      patience = 50  # Number of epochs to wait for improvement
      epochs_no_improve = 0

      # Training loop
      # print("Starting training...")
      for epoch in range(num_epochs):
          model.train()
          for i, (sequences, targets) in enumerate(train_data):
              # Move data to GPU if available
              sequences = sequences.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
              targets = targets.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

              # Forward pass
              outputs = model(sequences)
              loss = criterion(outputs, targets)

              # Backward and optimize
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

          # Print loss every few epochs
          if (epoch + 1) % 50 == 0:
              print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

          # Evaluation on the evaluation set
          model.eval()
          with torch.no_grad():
              eval_loss = 0
              for sequences, targets in eval_data:
                  # Move data to GPU if available
                  sequences = sequences.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                  targets = targets.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

                  outputs = model(sequences)
                  eval_loss += criterion(outputs, targets).item()

              avg_eval_loss = eval_loss / len(eval_data)

          # Early stopping check
          if avg_eval_loss < best_eval_loss:
              best_eval_loss = avg_eval_loss
              epochs_no_improve = 0
              # Optionally save the best model state
              # torch.save(model.state_dict(), 'best_model.pth')
          else:
              epochs_no_improve += 1
              if epochs_no_improve == patience:
                  # print(f'Early stopping at epoch {epoch+1}')
                  break # Stop training loop

      print("Training finished.")

      # Evaluation on the test set
      model.eval()
      with torch.no_grad():
          test_loss = 0
          for sequences, targets in test_data:
              # Move data to GPU if available
              sequences = sequences.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
              targets = targets.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

              outputs = model(sequences)
              test_loss += eval_function(outputs, targets).item()

          avg_test_loss = test_loss / len(test_data)
          print(f'{model_name} Test Loss: {avg_test_loss:.4f}')
          test_losses.append(avg_test_loss)

  return test_losses