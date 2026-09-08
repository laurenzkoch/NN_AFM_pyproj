from src.modules.simulation.pipeline import SimulationPipeline
from src.modules.generators.machine_generator import AFMMachineGenerator


def main():

    j_max = 5
    i_max = 5
    k_max = 5

    #sum_ = 0
    #for i in range(1, i_max + 1):
    #    for j in range(1, j_max + 1):
    #        for k in range(1, k_max + 1):
    #            sum_ += i * j * k * 5**3
    #            generator = AFMMachineGenerator(
    #                num_base_designs=i,
    #                num_sub_designs=j,
    #                num_subsub_designs=k,
    #                num_operation_points_per_dim=2
    #            )
    #            generator.geometry_test()
    #print("Total number of datapoints:", sum_)

    counter = 0
    for i in range(1, i_max + 1):
        for j in range(1, j_max + 1):
            for k in range(1, k_max + 1):
                with SimulationPipeline(
                     num_base_designs=i,
                     num_sub_designs=j,
                     num_subsub_designs=k,
                     num_operation_points_per_dim=5,
                     num_kernel=4,
                     autostart=True,
                     start_eduvpn=False
                ) as pipeline:
                    counter += 1
                    print(f"### {counter} of {i_max*j_max*k_max} ###")



if __name__ == "__main__":
    main()