from src.modules.simulation.sim_pipeline import SimulationPipeline


def main():
    pipeline = SimulationPipeline(
        num_base_designs=1,
        num_lam_designs=30,
        start_eduvpn=False,
        gen_type="v2",
        # csv_path="Results_EMagnetic_simulation_1_10_5_GenType-v2_30.09.2026_15-08-54_CEST.csv",
    )

    pipeline.start()


if __name__ == "__main__":
    main()
