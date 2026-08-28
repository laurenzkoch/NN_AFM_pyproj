from V2.modules.generators import data_generating_pipeline


def main():

    with data_generating_pipeline.MCADDataGenerator() as data_generator:
        data_generator.start(
            num_base_designs=4,
            num_sub_designs=3,
            num_subsub_designs=2,
            num_operation_points=100,
            num_kernel=4
        )



if __name__ == "__main__":
    main()