#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// structures and functions for data load
typedef struct data_structure {
    unsigned magic_number;
    unsigned data_count;
    unsigned height, length;
    double **data;
} data_structure;

typedef struct label_structure {
    unsigned magic_number;
    unsigned data_count;
    double **data;
} label_structure;

unsigned reverse_int(unsigned rint) {
    return (unsigned) (((rint << 24) & 0xff000000) | ((rint << 8) & 0x00ff0000) |
                       ((rint >> 8) & 0x0000ff00) | ((rint >> 24) & 0x000000ff));
}

data_structure *load_data(const char *file_name) {
    FILE *data_file = fopen(file_name, "rb");
    data_structure *new_data_structure = (data_structure *) malloc(sizeof(data_structure));
    fread(new_data_structure, 4, 4, data_file);
    new_data_structure->data_count = reverse_int(new_data_structure->data_count);
    new_data_structure->height = reverse_int(new_data_structure->height);
    new_data_structure->length = reverse_int(new_data_structure->length);
    printf("load data file(%s) : data count = %u height = %u length = %u dimension = %d\n",
           file_name, new_data_structure->data_count, new_data_structure->height, new_data_structure->length,
           new_data_structure->height * new_data_structure->length);
    new_data_structure->data = (double **) malloc(sizeof(double *) * new_data_structure->data_count);
    unsigned data_depth = new_data_structure->height * new_data_structure->length;

    char *tmp = (char *) malloc(sizeof(char) * data_depth);

    for (int i = 0; i < new_data_structure->data_count; ++i) {
        new_data_structure->data[i] = (double *) malloc(data_depth * sizeof(double));
        fread(tmp, 1, data_depth, data_file);
        for (int j = 0; j < data_depth; ++j) {
            new_data_structure->data[i][j] = ((double) tmp[j]) / 128;
        }
    }
    fclose(data_file);
    return new_data_structure;
}

label_structure *load_label(const char *file_name) {
    FILE *label_file = fopen(file_name, "rb");
    label_structure *new_label_structure = (label_structure *) malloc(sizeof(label_structure));
    fread(new_label_structure, 4, 2, label_file);
    new_label_structure->data_count = reverse_int(new_label_structure->data_count);
    printf("load label file(%s) : data count = %u\n", file_name, new_label_structure->data_count);
    new_label_structure->data = (double **) malloc(sizeof(double *) * new_label_structure->data_count);
    char *tmp = (char *) malloc(new_label_structure->data_count);
    fread(tmp, 1, new_label_structure->data_count, label_file);
    for (int i = 0; i < new_label_structure->data_count; ++i) {
        new_label_structure->data[i] = (double *) calloc(10, sizeof(double));
        new_label_structure->data[i][tmp[i]] = 1.0;
    }
    free(tmp);
    fclose(label_file);
    return new_label_structure;
}


// a neuro unit
typedef struct {
    double output_derivative;
    double input_derivative;
    double *weights;
    double *weights_derivative;
    double value;
} neuro_unit;

// a layer of neuro unit
typedef struct {
    int unit_count;
    int next_layer_unit_count;
    int pre_layer_unit_count;
    neuro_unit **unit;
} neuro_layer;

// a neuro network
typedef struct {
    int layer_count;
    neuro_layer **layer;
} neuro_network;
// init a neuro network
neuro_network *build_net_work(int layer_count, int *layer_unit_count) {
    if (layer_count < 2) printf("layer count error.\n");
    for (int i = 0; i < layer_count; ++i) {
        if (layer_unit_count[i] <= 0) printf("layer unit count at layer %d error.\n", i);
    }
    neuro_network *new_neuro_net_work = (neuro_network *) malloc(sizeof(neuro_network));
    new_neuro_net_work->layer = (neuro_layer **) malloc(sizeof(neuro_layer *) * layer_count);
    new_neuro_net_work->layer_count = layer_count;
    // first_layer
    new_neuro_net_work->layer[0] = (neuro_layer *) malloc(sizeof(neuro_layer));
    new_neuro_net_work->layer[0]->pre_layer_unit_count = 0;
    new_neuro_net_work->layer[0]->unit_count = layer_unit_count[0];
    new_neuro_net_work->layer[0]->next_layer_unit_count = layer_unit_count[1];
    new_neuro_net_work->layer[0]->unit = (neuro_unit **) malloc(sizeof(neuro_unit *) * layer_unit_count[0]);
    // hidden_layer
    for (int i = 1; i < layer_count - 1; ++i) {
        new_neuro_net_work->layer[i] = (neuro_layer *) malloc(sizeof(neuro_layer));
        new_neuro_net_work->layer[i]->pre_layer_unit_count = layer_unit_count[i - 1];
        new_neuro_net_work->layer[i]->unit_count = layer_unit_count[i];
        new_neuro_net_work->layer[i]->next_layer_unit_count = layer_unit_count[i + 1];
        new_neuro_net_work->layer[i]->unit = (neuro_unit **) malloc(sizeof(neuro_unit *) * layer_unit_count[i]);
    }
    // last_layer
    new_neuro_net_work->layer[layer_count - 1] = (neuro_layer *) malloc(sizeof(neuro_layer));
    new_neuro_net_work->layer[layer_count - 1]->pre_layer_unit_count = layer_unit_count[layer_count - 2];
    new_neuro_net_work->layer[layer_count - 1]->unit_count = layer_unit_count[layer_count - 1];
    new_neuro_net_work->layer[layer_count - 1]->next_layer_unit_count = 0;
    new_neuro_net_work->layer[layer_count - 1]->unit =
            (neuro_unit **) malloc(sizeof(neuro_unit *) * layer_unit_count[layer_count - 1]);
    // build weight
    for (int i = 0; i < layer_count; ++i) {
        int next_layer = new_neuro_net_work->layer[i]->next_layer_unit_count;
        for (int j = 0; j < layer_unit_count[i]; ++j) {
            new_neuro_net_work->layer[i]->unit[j] = (neuro_unit *) malloc(sizeof(neuro_unit));
            new_neuro_net_work->layer[i]->unit[j]->output_derivative = 0;
            new_neuro_net_work->layer[i]->unit[j]->input_derivative = 0;
            new_neuro_net_work->layer[i]->unit[j]->value = 0;
            new_neuro_net_work->layer[i]->unit[j]->weights =
                    (double *) malloc(sizeof(double) * next_layer);
            for (int k = 0; k < next_layer; ++k) {
                new_neuro_net_work->layer[i]->unit[j]->weights[k] = -1.0 + 2 * (((double) rand()) / RAND_MAX);
            }
            new_neuro_net_work->layer[i]->unit[j]->weights_derivative =
                    (double *) malloc(sizeof(double) * next_layer);
            memset(new_neuro_net_work->layer[i]->unit[j]->weights_derivative, 0, sizeof(double) * next_layer);
        }
    }
    return new_neuro_net_work;
}

// sigmoid function
double sigmoid(double ix) {
    return 1.0 / (1.0 + exp(-ix));
}

// derivative of sigmoid function
double derivative_sigmoid(double ix) {
    return sigmoid(ix) * (1.0 - sigmoid(ix));
}

// forward propagation
void forward_propagation(neuro_network *nn, double *data) {
    // write input
    for (int i = 0; i < nn->layer[0]->unit_count; ++i) {
        nn->layer[0]->unit[i]->value = data[i];
    }
    // propagation
    for (int i = 1; i < nn->layer_count; ++i) {
        for (int j = 0; j < nn->layer[i]->unit_count; ++j) {
            double sum = 0;
            for (int k = 0; k < nn->layer[i]->pre_layer_unit_count; ++k) {
                sum += nn->layer[i - 1]->unit[k]->value * nn->layer[i - 1]->unit[k]->weights[j];
            }
            nn->layer[i]->unit[j]->value = sigmoid(sum);
        }
    }
}

void backward_propagation(neuro_network *nn, double *target) {
    // last layer
    for (int i = 0; i < nn->layer[nn->layer_count - 1]->unit_count; ++i) {
        // solve out derivative
        nn->layer[nn->layer_count - 1]->unit[i]->output_derivative =
                nn->layer[nn->layer_count - 1]->unit[i]->value - target[i];
        // solve in derivative
        nn->layer[nn->layer_count - 1]->unit[i]->input_derivative =
                derivative_sigmoid(nn->layer[nn->layer_count - 1]->unit[i]->value) *
                nn->layer[nn->layer_count - 1]->unit[i]->output_derivative;
    }
    // hidden layer
    for (int i = nn->layer_count - 2; i >= 0; --i) {
        for (int j = 0; j < nn->layer[i]->unit_count; ++j) {
            // solve weight derivative (error sum up)
            for (int k = 0; k < nn->layer[i]->next_layer_unit_count; ++k) {
                nn->layer[i]->unit[j]->weights_derivative[k] +=
                        nn->layer[i + 1]->unit[k]->input_derivative * nn->layer[i]->unit[j]->value;
            }
            // solve out derivative
            double output_der = 0.0;
            for (int k = 0; k < nn->layer[i]->next_layer_unit_count; ++k) {
                output_der += nn->layer[i + 1]->unit[k]->input_derivative *
                              nn->layer[i]->unit[j]->weights[k];
            }
            nn->layer[i]->unit[j]->output_derivative = output_der;
            //solve in derivative
            nn->layer[i]->unit[j]->input_derivative =
                    derivative_sigmoid(nn->layer[i]->unit[j]->value) * nn->layer[i]->unit[j]->output_derivative;
        }
    }
}

void clean_network(neuro_network *nn) {
    for (int i = 0; i < nn->layer_count; ++i) {
        for (int j = 0; j < nn->layer[i]->unit_count; ++j) {
            memset(nn->layer[i]->unit[j]->weights_derivative, 0, sizeof(double) * nn->layer[i]->next_layer_unit_count);
        }
    }
}


double fix_error_value(neuro_network *nn, double alpha) {
    double fixed_val = 0;
    for (int i = 0; i < nn->layer_count - 1; ++i) {
        for (int j = 0; j < nn->layer[i]->unit_count; ++j) {
            for (int k = 0; k < nn->layer[i]->next_layer_unit_count; ++k) {
                fixed_val +=
                        nn->layer[i]->unit[j]->weights_derivative[k] * nn->layer[i]->unit[j]->weights_derivative[k];
                nn->layer[i]->unit[j]->weights[k] -= alpha * nn->layer[i]->unit[j]->weights_derivative[k];
            }
        }
    }
    return sqrt(fixed_val);
}

typedef struct {
    double alpha_factor;
    double multiple_factor;
    double allow_error;
    int max_train_times;
} factor_structure;

factor_structure *
write_factor(double alpha, double mult, double allowerr, int train_times) {
    factor_structure *new_factor_structure = (factor_structure *) malloc(sizeof(factor_structure));
    new_factor_structure->alpha_factor = alpha;
    new_factor_structure->multiple_factor = mult;
    new_factor_structure->allow_error = allowerr;
    new_factor_structure->max_train_times = train_times;
    return new_factor_structure;
}

void save_network(const char *filename, neuro_network *nn) {
    FILE *nnfile = fopen(filename, "wb");
    fwrite(&nn->layer_count, 4, 1, nnfile);
    for (int i = 0; i < nn->layer_count; ++i) {
        fwrite(&nn->layer[i]->unit_count, 4, 1, nnfile);
        fwrite(&nn->layer[i]->pre_layer_unit_count, 4, 1, nnfile);
        fwrite(&nn->layer[i]->next_layer_unit_count, 4, 1, nnfile);
        for (int j = 0; j < nn->layer[i]->unit_count; ++j) {
            fwrite(nn->layer[i]->unit[j]->weights, sizeof(double),
                   nn->layer[i]->next_layer_unit_count, nnfile);
            fwrite(nn->layer[i]->unit[j]->weights_derivative, sizeof(double),
                   nn->layer[i]->next_layer_unit_count, nnfile);
        }
    }
    fclose(nnfile);
    printf("complete save network : filename = %s.\n", filename);
}

neuro_network *load_network(const char *filename) {
    FILE *nnfile = fopen(filename, "rb");
    neuro_network *nn = (neuro_network *) malloc(sizeof(neuro_network));
    fread(&nn->layer_count, 4, 1, nnfile);
    nn->layer = (neuro_layer **) malloc(sizeof(neuro_layer *) * nn->layer_count);
    for (int i = 0; i < nn->layer_count; ++i) {
        nn->layer[i] = (neuro_layer *) malloc(sizeof(neuro_layer));
        fread(&nn->layer[i]->unit_count, 4, 1, nnfile);
        fread(&nn->layer[i]->pre_layer_unit_count, 4, 1, nnfile);
        fread(&nn->layer[i]->next_layer_unit_count, 4, 1, nnfile);
        nn->layer[i]->unit = (neuro_unit **) malloc(sizeof(neuro_unit *) * nn->layer[i]->unit_count);
        for (int j = 0; j < nn->layer[i]->unit_count; ++j) {
            nn->layer[i]->unit[j] = (neuro_unit *) malloc(sizeof(neuro_unit));
            nn->layer[i]->unit[j]->weights = (double *) malloc(sizeof(double) * nn->layer[i]->unit_count);
            fread(nn->layer[i]->unit[j]->weights, sizeof(double),
                  nn->layer[i]->next_layer_unit_count, nnfile);
            nn->layer[i]->unit[j]->weights_derivative = (double *) malloc(sizeof(double) * nn->layer[i]->unit_count);
            fread(nn->layer[i]->unit[j]->weights_derivative, sizeof(double),
                  nn->layer[i]->next_layer_unit_count, nnfile);
        }
    }
    fclose(nnfile);
    printf("load network file(%s).\n", filename);
    return nn;
}

/*
void pick(int range, int count, int *pic) {
    for (int i = 0; i < count; ++i) {
        pic[i] = (int) ((range - 1) * ((double) rand() / RAND_MAX));
    }
}
 */

// train network
void
train_network(neuro_network *nn, data_structure *train_data, label_structure *train_label, factor_structure *factors) {
    srand(44448763);
    // initial network
    // neuro_network *nn = build_net_work(factors->layer_count, factors->unit_count);
    double current_alpha = factors->alpha_factor;
    double multiple_factor = factors->multiple_factor;
    double allowed_error = factors->allow_error;
    int max_training_times = factors->max_train_times;
    double error_length = HUGE_VAL;
    for (int i = 0; i < max_training_times; ++i) {
        for (int j = 0; j < train_data->data_count; ++j) {
            forward_propagation(nn, train_data->data[j]);
            backward_propagation(nn, train_label->data[j]);
        }
        error_length = fix_error_value(nn, current_alpha) / sqrt(train_data->data_count);
        current_alpha = multiple_factor * error_length;
        printf("error_length(RMS) = %0.15lf , alpha_factor(Eta) = %0.15f\n", error_length, current_alpha);
        clean_network(nn);
        if (error_length < allowed_error) break;
    }
}

// test network validation
void test_validation(neuro_network *nn, data_structure *test_data, label_structure *test_label) {
    int judge_matrix[10][10] = {0};
    int valid = 0;
    for (int i = 0; i < test_data->data_count; ++i) {
        forward_propagation(nn, test_data->data[i]);
        int sel = 0, tar = 0;
        for (int j = 0; j < 10; ++j) {
            if (nn->layer[nn->layer_count - 1]->unit[j]->value > nn->layer[nn->layer_count - 1]->unit[sel]->value)
                sel = j;
            if (test_label->data[i][j] > test_label->data[i][tar]) tar = j;
        }
        judge_matrix[tar][sel]++;
        if (tar == sel) valid++;
    }
    printf("[ %d / %d ] validation. accuracy = %lf %.\n", valid, test_data->data_count,
           ((double) 100.0 * valid) / test_data->data_count);
    printf("Judge matrix\n\t0\t1\t2\t3\t4\t5\t6\t7\t8\t9\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d\t", i);
        for (int j = 0; j < 10; ++j) {
            printf("%d\t", judge_matrix[i][j]);
        }
        printf("\n");
    }
}

int main() {
    // for trainning
    data_structure *train_data = load_data("train_file");
    label_structure *train_label = load_label("train_label");

    // [784 20 10]
    // [0 0.0003 1.0 10000]
    factor_structure *facs = write_factor(0, 0.0003, 0.2, 10000);

    neuro_network *network = load_network("network7");
    train_network(network, train_data, train_label, facs);
    save_network("network8", network);

    // for testing
    data_structure *test_data = load_data("test_file");
    label_structure *test_label = load_label("test_label");
    test_validation(network, test_data, test_label);

    return 0;
}
