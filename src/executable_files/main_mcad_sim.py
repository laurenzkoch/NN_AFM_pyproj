from src.modules.simulation.pipeline import SimulationPipeline
from src.modules.generators.machine_generator import AFMMachineGenerator


def main():

    i_max = 5
    j_max = 5

    #sum_ = 0
    #for i in range(3, i_max + 1):
    #    for j in range(3, j_max + 1):
    #            sum_ += i * j**3 * 5**3
    #            with AFMMachineGenerator(
    #                num_base_designs=i,
    #                num_lam_designs=j,
    #                num_operation_points_per_dim=5
    #           ) as generator:
    #                generator.geometry_test()

    #print("Total number of datapoints:", sum_)

    counter = 0
    for i in range(3, i_max + 1):
        for j in range(3, j_max + 1):
                with SimulationPipeline(
                     num_base_designs=i,
                     num_lam_designs=j,
                     num_operation_points_per_dim=5,
                     num_kernel=4,
                     autostart=True,
                     start_eduvpn=False
                ) as pipeline:
                    counter += 1
                    print(f"### {counter} of {i_max*j_max} ###")



if __name__ == "__main__":
    main()