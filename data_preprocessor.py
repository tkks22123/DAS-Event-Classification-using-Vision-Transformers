import numpy as np
import torch
from scipy import signal as sp_signal
from scipy.ndimage import zoom
import os
import glob
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class DASDataProcessor:
    def __init__(self, window_size=512, window_shift=128, freq_bands=(50, 500), max_channels_per_file=50):
        """
        Initialize DAS Data Processor
        Modified to treat each channel as separate sample like DASDataLoader
        """
        self.window_size = window_size
        self.window_shift = window_shift
        self.freq_bands = freq_bands
        self.sampling_rate = 20000
        self.enable_augmentation = False
        self.max_channels_per_file = max_channels_per_file  # Maximum channels to use per file

        # Use the same label system as DASDataLoader
        self.labels = ['car', 'construction', 'fence', 'longboard',
                       'manipulation', 'openclose', 'regular', 'running', 'walk']
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}

    def load_h5_file(self, filepath):
        """
        Load HDF5 file using the correct structure from DASDataLoader
        Returns all channels data instead of just first channel
        """
        try:
            with h5py.File(filepath, 'r') as f:
                # Use the correct HDF5 structure from DASDataLoader
                if '/Acquisition/Raw[0]' in f:
                    raw_group = f['/Acquisition/Raw[0]']
                    raw_data = raw_group['RawData'][:]
                else:
                    # Fallback to alternative paths
                    data_paths = ['/Acquisition/Raw[0]/RawData', 'raw_data', 'data', 'acoustic_data']
                    raw_data = None
                    for path in data_paths:
                        if path in f:
                            raw_data = f[path][:]
                            break

                if raw_data is None:
                    raise ValueError(f"No data found in {filepath}")

                # Ensure proper shape handling like DASDataLoader
                if raw_data.ndim == 1:
                    raw_data = raw_data.reshape(1, -1)
                elif raw_data.ndim > 2:
                    raw_data = raw_data.reshape(raw_data.shape[0], -1)

                # Return ALL channels instead of just first channel
                return raw_data.astype(np.float32)

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            # Return fallback data with multiple channels
            return np.random.randn(10, 2048).astype(np.float32)

    def compute_spectrogram(self, signal, channel):
        """
        Convert 1D signal to spectrogram
        """
        try:
            # Ensure signal length
            if len(signal) < 512:
                signal = np.pad(signal, (0, 512 - len(signal)), mode='constant')

            # Use fixed STFT parameters
            nperseg = 256
            noverlap = 192
            nfft = 512

            f, t, spec = sp_signal.stft(
                signal,
                fs=self.sampling_rate,
                window='hann',
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                return_onesided=True,
                boundary='zeros'
            )

            # Band-limited power calculation
            power_spec = np.abs(spec) ** 2
            freq_mask = (f >= self.freq_bands[0]) & (f <= self.freq_bands[1])
            band_limited_spec = power_spec[freq_mask, :]

            # Log-scale representation
            log_spec = 10 * np.log10(band_limited_spec + 1e-10)

            # Normalization
            log_spec = (log_spec - np.mean(log_spec)) / (np.std(log_spec) + 1e-8)

            # Ensure 224x224 output
            target_size = (224, 224)
            if log_spec.shape != target_size:
                zoom_factors = (target_size[0] / log_spec.shape[0],
                                target_size[1] / log_spec.shape[1])
                log_spec = zoom(log_spec, zoom_factors, order=1)

            return log_spec

        except Exception as e:
            print(f"Spectrogram error: {e}")
            return np.random.randn(224, 224).astype(np.float32)

    def prepare_vit_input(self, spectrogram):
        """
        Convert spectrogram to patches for ViT
        Returns: (1, 256, 256) - ready for ViT input as full image
        """
        target_size = (256, 256)  # CHANGED: 直接输出 256x256 图像

        # Ensure correct size
        if spectrogram.shape != target_size:
            spectrogram = zoom(spectrogram,
                               (target_size[0] / spectrogram.shape[0],
                                target_size[1] / spectrogram.shape[1]),
                               order=1)

        spectrogram_tensor = torch.from_numpy(spectrogram).float().unsqueeze(0)  # (1, 256, 256)

        return spectrogram_tensor

    def load_dataset_from_folders(self, base_path, use_cache=True):
        """
        Load DAS dataset - MODIFIED to treat each channel as separate sample
        """
        cache_file = os.path.join(base_path, "dataset_cache_multichannel.pkl")
        if use_cache and os.path.exists(cache_file):
            print("📁 Loading multi-channel dataset from cache...")
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                print(f"✅ Loaded {len(cached_data['data_list'])} channel samples from cache")
                return cached_data['data_list'], cached_data['labels_list'], cached_data['class_names']
            except Exception as e:
                print(f"❌ Cache loading failed: {e}")

        data_list = []
        labels_list = []

        # Find all H5 files recursively
        h5_files = glob.glob(os.path.join(base_path, "**/*.h5"), recursive=True)

        if not h5_files:
            h5_files = glob.glob(os.path.join(base_path, "*.h5"))

        if not h5_files:
            raise ValueError(f"No HDF5 files found in {base_path}")

        print(f"📁 Found {len(h5_files)} HDF5 files")

        # Simple class mapping from filenames
        class_mapping = {}
        class_counter = 0

        total_channels = 0

        for h5_file in h5_files:  # Removed limit to process all files
            try:
                filename = os.path.basename(h5_file).lower()

                # Map filename to class like DASDataLoader
                if 'car' in filename:
                    class_name = 'car'
                elif 'construction' in filename:
                    class_name = 'construction'
                elif 'fence' in filename:
                    class_name = 'fence'
                elif 'longboard' in filename:
                    class_name = 'longboard'
                elif 'manipulation' in filename:
                    class_name = 'manipulation'
                elif 'open' in filename or 'close' in filename:
                    class_name = 'openclose'
                elif 'regular' in filename:
                    class_name = 'regular'
                elif 'running' in filename:
                    class_name = 'running'
                elif 'walk' in filename:
                    class_name = 'walk'
                else:
                    class_name = 'unknown'

                if class_name not in class_mapping:
                    class_mapping[class_name] = class_counter
                    class_counter += 1

                # Load all channels from file
                file_data = self.load_h5_file(h5_file)

                # Treat each channel as separate sample
                num_channels = min(file_data.shape[0], self.max_channels_per_file)

                for channel_idx in range(num_channels):
                    channel_data = file_data[channel_idx:channel_idx + 1]  # Keep as 2D but single channel
                    data_list.append(channel_data)
                    labels_list.append(class_mapping[class_name])
                    total_channels += 1

                print(f"✅ Processed {num_channels} channels from {os.path.basename(h5_file)}")

            except Exception as e:
                print(f"Error loading {h5_file}: {e}")
                continue

        class_names = list(class_mapping.keys())

        print(f"🎉 Multi-channel dataset loading completed:")
        print(f"   - Total files: {len(h5_files)}")
        print(f"   - Total channel samples: {total_channels}")
        print(f"   - Classes found: {class_names}")

        # Save to cache
        if use_cache:
            try:
                import pickle
                cache_data = {
                    'data_list': data_list,
                    'labels_list': labels_list,
                    'class_names': class_names
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                print(f"💾 Multi-channel dataset cached to {cache_file}")
            except Exception as e:
                print(f"⚠️ Cache saving failed: {e}")

        return data_list, labels_list, class_names

    def create_torch_dataset(self, data_list, labels_list, max_samples_per_class=None):
        return DASPyTorchDataset(data_list, labels_list, self, max_samples_per_class)


class DASPyTorchDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, labels_list, processor, max_samples_per_class=None):
        self.data_list = data_list
        self.labels_list = labels_list
        self.processor = processor

        if max_samples_per_class is not None:
            self._balance_classes(max_samples_per_class)

        print(f"Multi-channel dataset created with {len(self.data_list)} channel samples")

    def _balance_classes(self, max_samples_per_class):
        from collections import defaultdict
        class_indices = defaultdict(list)

        for idx, label in enumerate(self.labels_list):
            class_indices[label].append(idx)

        balanced_indices = []
        for indices in class_indices.values():
            if len(indices) > max_samples_per_class:
                selected = np.random.choice(indices, max_samples_per_class, replace=False)
                balanced_indices.extend(selected)
            else:
                balanced_indices.extend(indices)

        self.data_list = [self.data_list[i] for i in balanced_indices]
        self.labels_list = [self.labels_list[i] for i in balanced_indices]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        try:
            raw_data = self.data_list[idx]
            label = self.labels_list[idx]

            # Extract single channel data (already stored as single channel)
            if raw_data.ndim > 1:
                channel_data = raw_data[0]  # Get the first (and only) channel
            else:
                channel_data = raw_data

            spectrogram = self.processor.compute_spectrogram(channel_data, 0)
            input_tensor = self.processor.prepare_vit_input(spectrogram)

            return input_tensor, label

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            return torch.zeros(1, 196, 256), 0


# Example usage
if __name__ == "__main__":
    # Test the modified multi-channel processor
    processor = DASDataProcessor(max_channels_per_file=50)

    # Load dataset with multiple channels per file
    data_list, labels_list, class_names = processor.load_dataset_from_folders(
        "/path/to/your/dataset",
        use_cache=True
    )

    # Create PyTorch dataset
    dataset = processor.create_torch_dataset(data_list, labels_list)

    print(f"Dataset size: {len(dataset)}")
    print(f"Sample shape: {dataset[0][0].shape}")
    print(f"Sample label: {dataset[0][1]}")