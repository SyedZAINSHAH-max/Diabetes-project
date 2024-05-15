:- use_module(library(csv)).
:- use_module(library(random)).

% 1. Read the data.
read_data(File, Data) :-
    csv_read_file(File, Data, [functor(record), arity(13)]).

% 2. Make necessary corrections on the data. (Placeholder)
% Placeholder for data preprocessing
preprocess_data(Data, ProcessedData) :-
    % Drop irrelevant columns (ID and No_Pation)
    drop_columns(Data, ['ID', 'No_Pation'], Data1),
    % Encode categorical variable (Gender) to numerical
    encode_categorical(Data1, 'Gender', ProcessedData).

% Predicate to drop columns from the dataset
drop_columns([], _, []).
drop_columns([Row|Rows], ColumnsToDrop, [NewRow|NewRows]) :-
    drop_columns_helper(Row, ColumnsToDrop, NewRow),
    drop_columns(Rows, ColumnsToDrop, NewRows).

drop_columns_helper(Row, [], Row).
drop_columns_helper(Row, [Column|Columns], NewRow) :-
    select(Column=_Value, Row, _Rest),
    drop_columns_helper(Row, Columns, NewRow).

% Predicate to encode categorical variable to numerical
encode_categorical([], _, []).
encode_categorical([Row|Rows], Column, [NewRow|NewRows]) :-
    encode_categorical_helper(Row, Column, NewValue),
    NewRow = [Column=NewValue|Rest],
    encode_categorical(Rows, Column, NewRows).

encode_categorical_helper(Row, Column, NewValue) :-
    member(Column=OriginalValue, Row),
    (   OriginalValue = 'F' -> NewValue = 0
    ;   OriginalValue = 'M' -> NewValue = 1
    ;   NewValue = OriginalValue
    ).


% 3. Create training and testing datasets.
split_data(Data, Train, Test) :-
    random_permutation(Data, Shuffled),
    length(Shuffled, Len),
    TrainLen is round(0.8 * Len),
    append(Train, Test, Shuffled),
    length(Train, TrainLen).

% Placeholder for building and training the neural network
build_and_train_neural_network :-
    % Call Python script to build and train the neural network
    shell('python diabetes_neural_network.py', _).

% Placeholder for evaluating the neural network
evaluate_neural_network :-
    % Call Python script to evaluate the neural network
    shell('python evaluate_neural_network.py', _).

% Example confusion matrix (Placeholder)
example_confusion_matrix(confusion_matrix(diabetic, diabetic, 10),
                         confusion_matrix(diabetic, non_diabetic, 5),
                         confusion_matrix(non_diabetic, diabetic, 3),
                         confusion_matrix(non_diabetic, non_diabetic, 20)).

% Main predicate to demonstrate the process
main :-
    % Read data from CSV file
    read_data('Dataset of Diabetes.csv', Data),
    
    % Make necessary corrections on the data (Placeholder)
    % You can implement data preprocessing here in Prolog or in another language.
    
    % Create training and testing datasets
    split_data(Data, Train, Test),
    
    % Build and train the neural network
    build_and_train_neural_network,
    
    % Evaluate the neural network
    evaluate_neural_network,
    
    % Placeholder for evaluating the neural network
    % Calculate accuracy and precision using a confusion matrix
    example_confusion_matrix(ConfusionMatrix),
    calculate_metrics(ConfusionMatrix, Accuracy, Precision),
    format('Accuracy: ~2f%~n', [Accuracy * 100]),
    format('Precision: ~2f%~n', [Precision * 100]).

% Run the main predicate
:- initialization(main).
