from V1.modules.generators import data_generating_main_investigation


def main():

    # "Winding", "Axial", "Operational Point", "Radial", "Test Points"
    with data_generating_main_investigation.MCADDataGenerator() as data_generator:
        data_generator.start(
            variation_type="Operational Point",
            number_of_datapoints=2000,
            num_kernel=4
        )



if __name__ == "__main__":
    main()