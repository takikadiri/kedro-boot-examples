import mlflow
from kedro_boot.app import AbstractKedroBootApp
from kedro_boot.session import KedroBootSession


class MonteCarloApp(AbstractKedroBootApp):
    def _run(self, kedro_boot_session: KedroBootSession):

        # leveraging config_loader to manage app's configs
        monte_carlo_params = kedro_boot_session.config_loader.get("monte_carlo.yml")
        sample_range = monte_carlo_params["sample_range"]
        radius = monte_carlo_params["radius"]

        for num_samples in range(
            sample_range["start"], sample_range["stop"], sample_range["step"]
        ):
            distances = []
            with mlflow.start_run(nested=True):
                for _ in range(num_samples):
                    distance = kedro_boot_session.run(
                        name="simulate_distance", parameters={"radius": radius}
                    )
                    distances.append(distance)

                estimated_pi = kedro_boot_session.run(
                    name="estimate_pi",
                    inputs={"distances": distances},
                    parameters={"num_samples": num_samples, "radius": radius},
                )

            # Log estimated pi in the mlflow parent run
            mlflow.log_metric("estimated_pi", estimated_pi, num_samples)