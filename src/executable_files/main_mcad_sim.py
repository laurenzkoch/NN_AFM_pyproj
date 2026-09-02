from modules.simulation.pipeline import SimulationPipeline


def main():

    with SimulationPipeline(
            num_base_designs=5,
            num_sub_designs=5,
            num_subsub_designs=5,
            num_operation_points_per_dim=5,
            num_kernel=4
    ) as pipeline:
        pass



if __name__ == "__main__":
    main()