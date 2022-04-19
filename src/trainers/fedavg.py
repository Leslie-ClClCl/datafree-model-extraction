from ..trainers.BaseTrainer import BaseTrainer
from ..models.model import choose_model
from ..optimizer.gd import GD


class FedAvgTrainer(BaseTrainer):
    def __init__(self, options, dataset):
        model = choose_model(options)
        self.move_model_to_gpu(model)

        self.optimizer = GD(model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        super(FedAvgTrainer, self).__init__(dataset, options, model, self.optimizer)

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()

        for round_i in range(self.num_round):

            # Test latest model on train data
            # self.test_latest_model_on_traindata(round_i)
            self.test_latest_model_on_evaldata(round_i)

            # Choose K clients prop to data size
            selected_clients = self.select_clients(seed=round_i)

            # Solve minimization locally
            solns = self.local_train(round_i, selected_clients)

            # # Track communication cost
            # self.metrics.extend_commu_stats(round_i)

            # Update latest model
            self.latest_model = self.aggregate(solns)
            self.optimizer.inverse_prop_decay_learning_rate(round_i)

        # Test final model on train data
        # self.test_latest_model_on_traindata(self.num_round)
        self.test_latest_model_on_evaldata(self.num_round)

        # Save tracked information
        self.metrics.write()
