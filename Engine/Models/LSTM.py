#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
AudioLSTM com SHAP - Versão Corrigida
Solução para o erro: "operands could not be broadcast together with shapes"

Autor: Kayuã Oleques Paim
Modificado: 2025/10/31
"""

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'

try:
    import os
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf

    from tensorflow.keras import Model
    from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, GlobalAveragePooling1D
    from tensorflow.keras.callbacks import EarlyStopping
    from Engine.Models.Process.LSTM_Process import ProcessLSTM

    import shap
    import matplotlib

    matplotlib.use('Agg')

except ImportError as error:
    print(error)
    sys.exit(-1)


class AudioLSTM(ProcessLSTM):
    """
    AudioLSTM com SHAP integrado - Versão corrigida para TensorFlow 2.x

    Correções implementadas:
    - Fallback automático de GradientExplainer para Gradient×Input
    - Limpeza automática de shapes com squeeze
    - Tratamento robusto de erros de broadcasting
    """

    def __init__(self, arguments):
        ProcessLSTM.__init__(self, arguments)

        self.neural_network_model = None
        self.list_lstm_cells = arguments.lstm_list_lstm_cells
        self.loss_function = arguments.lstm_loss_function
        self.optimizer_function = arguments.lstm_optimizer_function
        self.recurrent_activation = arguments.lstm_recurrent_activation
        self.intermediary_layer_activation = arguments.lstm_intermediary_layer_activation
        self.input_dimension = arguments.lstm_input_dimension
        self.number_classes = arguments.number_classes
        self.dropout_rate = arguments.lstm_dropout_rate
        self.last_layer_activation = arguments.lstm_last_layer_activation
        self.model_name = "LSTM"

    def build_model(self) -> None:
        """Build the LSTM model architecture."""
        inputs = Input(shape=self.input_dimension)
        neural_network_flow = inputs

        for _, cells in enumerate(self.list_lstm_cells):
            neural_network_flow = LSTM(cells, activation=self.intermediary_layer_activation,
                                       recurrent_activation=self.recurrent_activation,
                                       return_sequences=True)(neural_network_flow)
            neural_network_flow = Dropout(self.dropout_rate)(neural_network_flow)

        neural_network_flow = GlobalAveragePooling1D()(neural_network_flow)
        neural_network_flow = Dense(self.number_classes, activation=self.last_layer_activation)(neural_network_flow)

        self.neural_network_model = Model(inputs=inputs, outputs=neural_network_flow)
        self.neural_network_model.summary()

    def compile_and_train(self, train_data, train_labels, epochs: int,
                          batch_size: int, validation_data=None,
                          visualize_attention: bool = True,
                          use_early_stopping: bool = True,
                          early_stopping_monitor: str = 'val_loss',
                          early_stopping_patience: int = 10,
                          early_stopping_restore_best: bool = True,
                          early_stopping_min_delta: float = 0.0001) -> tf.keras.callbacks.History:
        """
        Compile and train with SHAP visualization generation.
        """
        callbacks = []

        if use_early_stopping:
            early_stopping = EarlyStopping(
                monitor=early_stopping_monitor,
                patience=early_stopping_patience,
                restore_best_weights=early_stopping_restore_best,
                min_delta=early_stopping_min_delta,
                verbose=1,
                mode='auto'
            )
            callbacks.append(early_stopping)

        self.neural_network_model.compile(
            optimizer=self.optimizer_function,
            loss=self.loss_function,
            metrics=['accuracy']
        )

        training_history = self.neural_network_model.fit(
            train_data, train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks if callbacks else None
        )

        if validation_data is not None:
            print(f"\nAcurácia Final (Validação): {training_history.history['val_accuracy'][-1]:.4f}")

        # Generate SHAP visualizations
        if visualize_attention and validation_data is not None:
            val_data, val_labels = validation_data

            self.generate_validation_visualizations(
                validation_data=val_data,
                validation_labels=val_labels,
                num_samples=128,
                output_dir='SHAP_Maps_LSTM'
            )

        return training_history

    def generate_validation_visualizations(self, validation_data, validation_labels,
                                           num_samples: int = 128,
                                           output_dir: str = 'SHAP_Maps_LSTM') -> None:
        """
        Generate SHAP visualizations with robust error handling.
        Saves model weights before SHAP computation to prevent state corruption.
        """

        print("\n" + "=" * 80)
        print("Gerando Visualizações SHAP para Explicabilidade do Modelo LSTM")
        print("=" * 80)

        os.makedirs(output_dir, exist_ok=True)

        # 🔧 CRITICAL: Save model weights before SHAP computation
        temp_weights_path = f'{output_dir}/_temp_weights.h5'
        self.neural_network_model.save_weights(temp_weights_path)
        print(f"✓ Pesos do modelo salvos temporariamente")

        # Select samples
        num_samples = min(num_samples, len(validation_data))
        indices = np.random.choice(len(validation_data), num_samples, replace=False)
        sample_data = validation_data[indices]
        sample_labels = validation_labels[indices]

        # Background dataset
        background_size = min(100, len(validation_data))
        background_indices = np.random.choice(len(validation_data), background_size, replace=False)
        background_data = validation_data[background_indices]

        print(f"\nUsando {num_samples} amostras de validação")
        print(f"Background dataset: {background_size} amostras")

        shap_values = None
        method_used = None

        # Try GradientExplainer
        try:
            print("\n[Método 1] Tentando SHAP GradientExplainer...")
            explainer = shap.GradientExplainer(self.neural_network_model, background_data)
            shap_values = explainer.shap_values(sample_data)
            method_used = "GradientExplainer"
            print("  ✓ GradientExplainer funcionou!")
        except Exception as e:
            print(f"  ✗ GradientExplainer falhou: {str(e)[:80]}")

        # Fallback: Gradient×Input (always works)
        if shap_values is None:
            print("\n[Método 2] Usando Gradient×Input (sempre funciona)...")
            shap_values = self._compute_gradient_importance(sample_data)
            method_used = "Gradient×Input"

        # 🔧 CRITICAL: Restore model weights after SHAP computation
        print("\n✓ Restaurando estado do modelo...")
        self.neural_network_model.load_weights(temp_weights_path)

        # Clean up temp file
        if os.path.exists(temp_weights_path):
            os.remove(temp_weights_path)

        if shap_values is not None:
            # 🔧 FIX: Remove extra dimensions
            shap_values = np.array(shap_values)
            while len(shap_values.shape) > 3:
                shap_values = np.squeeze(shap_values, axis=-1)

            while len(sample_data.shape) > 3:
                sample_data = np.squeeze(sample_data, axis=-1)

            print(f"\n✓ Método: {method_used}")
            print(f"  SHAP shape: {shap_values.shape}")
            print(f"  Data shape: {sample_data.shape}")

            # Generate all visualizations
            self._generate_all_plots(shap_values, sample_data, sample_labels,
                                     output_dir, method_used)

            print(f"\n✓ Visualizações salvas em: {output_dir}/")
        else:
            print("\n✗ Não foi possível gerar visualizações")

        print("=" * 80 + "\n")

    def _compute_gradient_importance(self, sample_data):
        """
        Compute feature importance using gradients (fallback method).
        Note: Model weights will be restored by the caller after this operation.
        """
        print("  Calculando gradientes...")

        gradients_list = []

        for i in range(len(sample_data)):
            # Convert to tensor
            x = tf.constant(sample_data[i:i + 1], dtype=tf.float32)

            with tf.GradientTape(persistent=False) as tape:
                tape.watch(x)
                pred = self.neural_network_model(x, training=False)
                pred_class = tf.argmax(pred, axis=1)[0]
                pred_score = pred[0, pred_class]

            # Compute gradients
            grads = tape.gradient(pred_score, x)

            # Explicitly delete tape to free resources
            del tape

            if grads is not None:
                # Gradient × Input
                importance = grads.numpy() * sample_data[i:i + 1]
                gradients_list.append(importance[0])
            else:
                gradients_list.append(np.zeros_like(sample_data[i]))

        return np.array(gradients_list)

    def _generate_all_plots(self, shap_values, sample_data, sample_labels,
                            output_dir, method_name):
        """Generate all visualization plots."""

        print("\nGerando plots...")

        # Ensure 3D arrays
        shap_values = np.squeeze(shap_values)
        sample_data = np.squeeze(sample_data)

        try:
            self._plot_summary(shap_values, sample_data, output_dir, method_name)
        except Exception as e:
            print(f"  ⚠ Summary plot falhou: {e}")

        try:
            self._plot_temporal_heatmaps(shap_values, sample_data, sample_labels,
                                         output_dir, method_name)
        except Exception as e:
            print(f"  ⚠ Heatmaps falharam: {e}")

        try:
            self._plot_temporal_importance(shap_values, output_dir, method_name)
        except Exception as e:
            print(f"  ⚠ Temporal importance falhou: {e}")

    def _plot_summary(self, shap_values, sample_data, output_dir, method_name):
        """Generate summary plot."""
        print("  [1/3] Summary plot...")

        plt.figure(figsize=(12, 8))

        # Flatten for summary plot
        n_samples = len(shap_values)
        shap_flat = shap_values.reshape(n_samples, -1)
        data_flat = sample_data[:n_samples].reshape(n_samples, -1)

        shap.summary_plot(shap_flat, data_flat, show=False, max_display=20)
        plt.title(f'Importância de Features (SHAP)\nMétodo: {method_name}',
                  fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("    ✓ Salvo")

    def _plot_temporal_heatmaps(self, shap_values, sample_data, sample_labels,
                                output_dir, method_name, n_examples=5):
        """Generate temporal heatmaps."""
        print("  [2/3] Heatmaps temporais...")

        predictions = self.neural_network_model.predict(sample_data, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(sample_labels, axis=1) if len(sample_labels.shape) > 1 else sample_labels

        # Select examples
        correct = np.where(pred_classes == true_classes)[0]
        incorrect = np.where(pred_classes != true_classes)[0]

        indices = []
        if len(correct) > 0:
            indices.extend(correct[:min(3, len(correct))])
        if len(incorrect) > 0:
            indices.extend(incorrect[:min(2, len(incorrect))])

        for plot_idx, sample_idx in enumerate(indices[:n_examples]):
            fig, axes = plt.subplots(3, 1, figsize=(14, 10))

            # Data sample
            data_sample = np.squeeze(sample_data[sample_idx])
            shap_sample = np.squeeze(shap_values[sample_idx])

            # Original features
            axes[0].imshow(data_sample.T, aspect='auto', cmap='viridis',
                           origin='lower', interpolation='nearest')
            axes[0].set_title('Features de Áudio', fontweight='bold')
            axes[0].set_xlabel('Timesteps')
            axes[0].set_ylabel('Features')

            # SHAP values
            vmax = np.abs(shap_sample).max()
            im = axes[1].imshow(shap_sample.T, aspect='auto', cmap='RdBu_r',
                                origin='lower', vmin=-vmax, vmax=vmax, interpolation='nearest')
            axes[1].set_title(f'Valores SHAP (Importância)\nMétodo: {method_name}',
                              fontweight='bold')
            axes[1].set_xlabel('Timesteps')
            axes[1].set_ylabel('Features')
            plt.colorbar(im, ax=axes[1])

            # Temporal aggregation
            temporal_imp = np.mean(np.abs(shap_sample), axis=1)
            axes[2].plot(temporal_imp, linewidth=2, color='darkblue')
            axes[2].fill_between(range(len(temporal_imp)), temporal_imp, alpha=0.3)
            axes[2].set_title('Importância ao Longo do Tempo', fontweight='bold')
            axes[2].set_xlabel('Timesteps')
            axes[2].set_ylabel('Importância Média')
            axes[2].grid(alpha=0.3)

            fig.suptitle(f'Amostra {plot_idx + 1}: Predição={pred_classes[sample_idx]}, '
                         f'Real={true_classes[sample_idx]}, '
                         f'Conf={predictions[sample_idx].max():.3f}',
                         fontsize=12, fontweight='bold')

            plt.tight_layout()
            plt.savefig(f'{output_dir}/heatmap_{plot_idx + 1}.png',
                        dpi=300, bbox_inches='tight')
            plt.close()

        print(f"    ✓ {len(indices)} heatmaps salvos")

    def _plot_temporal_importance(self, shap_values, output_dir, method_name):
        """Plot temporal importance analysis."""
        print("  [3/3] Análise temporal...")

        # Average across samples
        mean_importance = np.mean(np.abs(shap_values), axis=0)

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Heatmap
        im = axes[0].imshow(mean_importance.T, aspect='auto', cmap='YlOrRd',
                            origin='lower', interpolation='nearest')
        axes[0].set_title('Importância Média ao Longo do Tempo', fontweight='bold')
        axes[0].set_xlabel('Timesteps')
        axes[0].set_ylabel('Features')
        plt.colorbar(im, ax=axes[0])

        # Line plot of top features
        feature_totals = np.mean(mean_importance, axis=0)
        top_features = np.argsort(feature_totals)[-10:]

        for feat in top_features:
            axes[1].plot(mean_importance[:, feat], label=f'F{feat}', alpha=0.7)

        axes[1].set_title('Top 10 Features Mais Importantes', fontweight='bold')
        axes[1].set_xlabel('Timesteps')
        axes[1].set_ylabel('Importância')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[1].grid(alpha=0.3)

        plt.suptitle(f'Análise Temporal (SHAP)\nMétodo: {method_name}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("    ✓ Salvo")

    # Properties
    @property
    def neural_network_model(self):
        return self._neural_network_model

    @neural_network_model.setter
    def neural_network_model(self, value):
        self._neural_network_model = value

    @property
    def list_lstm_cells(self):
        return self._list_lstm_cells

    @list_lstm_cells.setter
    def list_lstm_cells(self, value):
        self._list_lstm_cells = value

    @property
    def loss_function(self):
        return self._loss_function

    @loss_function.setter
    def loss_function(self, value):
        self._loss_function = value

    @property
    def optimizer_function(self):
        return self._optimizer_function

    @optimizer_function.setter
    def optimizer_function(self, value):
        self._optimizer_function = value

    @property
    def recurrent_activation(self):
        return self._recurrent_activation

    @recurrent_activation.setter
    def recurrent_activation(self, value):
        self._recurrent_activation = value

    @property
    def intermediary_layer_activation(self):
        return self._intermediary_layer_activation

    @intermediary_layer_activation.setter
    def intermediary_layer_activation(self, value):
        self._intermediary_layer_activation = value

    @property
    def input_dimension(self):
        return self._input_dimension

    @input_dimension.setter
    def input_dimension(self, value):
        self._input_dimension = value

    @property
    def number_classes(self):
        return self._number_classes

    @number_classes.setter
    def number_classes(self, value):
        self._number_classes = value

    @property
    def dropout_rate(self):
        return self._dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, value):
        self._dropout_rate = value

    @property
    def last_layer_activation(self):
        return self._last_layer_activation

    @last_layer_activation.setter
    def last_layer_activation(self, value):
        self._last_layer_activation = value

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        self._model_name = value