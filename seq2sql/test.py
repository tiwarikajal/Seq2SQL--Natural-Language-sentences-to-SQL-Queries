from util.utils import *
from util.constants import *
from seq2sql.model.seq2sql import Seq2SQL


def test_seq2sql():
    test_sql_data, test_table_data, TEST_DB = load_dataset("test")

    # load glove word embeddings and initialize the model
    word_emb = load_word_embeddings('glove/glove.6B.300d.txt')
    model = Seq2SQL(word_emb, N_word=300, gpu=GPU)

    # Load the best model state saved during training
    agg_m, sel_m, cond_m = best_model_name()
    model.agg_pred.load_state_dict(torch.load(agg_m))
    model.sel_pred.load_state_dict(torch.load(sel_m))
    model.cond_pred.load_state_dict(torch.load(cond_m))

    # Run the model on the test data and get the logical accuracy
    logical_accuracy_score =\
        epoch_acc(model, BATCH_SIZE, test_sql_data, test_table_data, save_results = True)

    # Run the model on the test data and get the execution accuracy
    execution_accuracy_score =\
        epoch_exec_acc(model, BATCH_SIZE, test_sql_data, test_table_data, TEST_DB)
    
    print("Test logical accuracy: %s;\n  breakdown on (agg, sel, where): %s" % logical_accuracy_score)
    print("Test execution accuracy: %s" % execution_accuracy_score)


if __name__ == '__main__':
    test_seq2sql()
