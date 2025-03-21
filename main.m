% Step 1: Load and preprocess data
data = readtable('Copy_of_dataset.txt', 'VariableNamingRule', 'preserve');

% Convert Date and Time to strings if needed
dateStr = string(data.Date); % Convert Date to string array
timeStr = string(data.Time); % Convert Time to string array

% Handle missing or invalid Date/Time values
validRows = ~ismissing(dateStr) & ~ismissing(timeStr);
data = data(validRows, :);
dateStr = dateStr(validRows);
timeStr = timeStr(validRows);

% Combine Date and Time into datetime
datetime_col = datetime(dateStr + " " + timeStr, 'InputFormat', 'dd/MM/yyyy HH:mm:ss');

% Extract features and target
X = data{:, ["Global_active_power", "Global_reactive_power", ...
 "Voltage", "Global_intensity", ...
 "Sub_metering_1", "Sub_metering_2", ...
 "Sub_metering_3"]};

% Remove rows with missing values in features or target
y = data.("Global_active_power"); % Adjust column name if target differs
validRows = all(~isnan(X), 2) & ~isnan(y);
X = X(validRows, :);
y = y(validRows);
datetime_col = datetime_col(validRows); % Filter datetime_col for valid rows

% Feature engineering: Extract hour, day, and month
X = [X, hour(datetime_col), day(datetime_col), month(datetime_col)];


% Step 2: Normalize the features and target manually

muX = mean(X, 1); % Mean for each feature
sigmaX = std(X, 0, 1); % Standard deviation for each feature
sigmaX(sigmaX == 0) = 1; % Avoid division by zero
X = (X - muX) ./ sigmaX; % Standardize features
muy = mean(y);
sigmay = std(y);
sigmay(sigmay == 0) = 1; % Avoid division by zero
y = (y - muy) ./ sigmay; % Standardize target


% Step 3: Split data into training and testing sets

cv = cvpartition(size(X, 1), 'Holdout', 0.2);
X_train = X(training(cv), :);
y_train = y(training(cv));
X_test = X(test(cv), :);
y_test = y(test(cv));

% Debug: Inspect data dimensions and values
disp('Data sizes:');
fprintf('X_train: %d x %d\n', size(X_train));
fprintf('y_train: %d\n', length(y_train));
fprintf('X_test: %d x %d\n', size(X_test));
fprintf('y_test: %d\n', length(y_test));


% Step 4: Define and configure neural network

hiddenLayerSize = 20; % Increased number of neurons
net = fitnet(hiddenLayerSize);
% Set network division (train, validation, test)
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

% Set training parameters
net.trainParam.epochs = 500; % Increased number of epochs
net.trainParam.showWindow = true; % Display training GUI

% Train the network
[net, tr] = train(net, X_train', y_train');
% Debug: Check training results
disp('Training results:');
disp(tr);


% Step 5: Evaluate the network


y_pred_train = net(X_train'); % Predict on training data
y_pred_test = net(X_test'); % Predict on testing data

% Denormalize the predictions and actual values
y_pred_train = y_pred_train' .* sigmay + muy; % Elementwise operation
y_pred_test = y_pred_test' .* sigmay + muy; % Elementwise operation
y_train = y_train .* sigmay + muy; % Elementwise operation
y_test = y_test .* sigmay + muy; % Elementwise operation

% Debug: Inspect predictions and actual values
disp('Sample predictions vs. actual values (test set):');
disp(table(y_pred_test(1:5), y_test(1:5), 'VariableNames', {'Predicted', 'Actual'}));


% Step 6: Performance Metrics

mae = mean(abs(y_test - y_pred_test));
rmse = sqrt(mean((y_test - y_pred_test).^2));
fprintf('MAE: %.2f kW\n', mae);
fprintf('RMSE: %.2f kW\n', rmse);


% Step 7: Visualizations

% 1. Network Diagram
figure;
view(net);

% 2. Performance Graph
figure;
plotperform(tr);

% 3. Training State
figure;
plottrainstate(tr);

% 4. Error Histogram
errors = y_test - y_pred_test; % Error vector
errors = errors(:); % Ensure it is a column vector
errors = errors(~isnan(errors) & ~isinf(errors)); % Clean errors
figure;
ploterrhist(errors);
title('Error Histogram');

% 5. Regression
figure;
plotregression(y_train, y_pred_train, 'Training Data', ...
 y_test, y_pred_test, 'Testing Data');

% 6. Time Series Response
figure;
plot(1:length(y_test), y_test, 'b', 'DisplayName', 'Actual Load');
hold on;
plot(1:length(y_pred_test), y_pred_test, 'r--', 'DisplayName', 'Predicted Load');
legend;
xlabel('Sample Index');
ylabel('Load (kW)');
title('Time Series Response');

% 7. Error Autocorrelation
figure;
autocorr(errors, min(50, length(errors) - 1)); % Plot autocorrelation with lag of 50

title('Error Autocorrelation');