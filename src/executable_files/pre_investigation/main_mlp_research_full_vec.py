

from modules.generators.NN_research import NNresearch
from modules.generators.data_generating_pre_investigation import BasicDataGenerator

def main():

    funcs = ["sin_exp", "linear"]
    basic_data_generator = BasicDataGenerator()
    researcher = NNresearch()

    for func in funcs:


        print(f"=======starting with {func} data quantity variation===================")

        data_points = [basic_data_generator.data_points(func=func, number_of_data_points=i) for i in range(50, 10010, 50)]
        xs = [data_points[i][["time"]] for i in range(len(data_points))]
        ys = [data_points[i][["y"]] for i in range(len(data_points))]

        researcher.full_vectorial_training(
            num_of_workers=4,
            min_num_of_layers=3,
            max_num_of_layers=3,
            min_neuron=100,
            max_neuron=101,
            number_neuron_variations=1,
            xs=xs,
            ys=ys,
            data_name=func
        )

        print(f"=======starting with {func} neuron variation===================")

        data_points = [basic_data_generator.data_points(func=func, number_of_data_points=1000)]
        xs = [data_points[i][["time"]] for i in range(len(data_points))]
        ys = [data_points[i][["y"]] for i in range(len(data_points))]
        researcher.full_vectorial_training(
            num_of_workers=4,
            max_num_of_layers=5,
            min_neuron=1,
            max_neuron=501,
            number_neuron_variations=1000,
            xs=xs,
            ys=ys,
            data_name=func
        )




if __name__ == "__main__":
    main()