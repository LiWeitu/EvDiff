# Public E2VID+ teacher

These model files are derived from the public
[EVREAL repository](https://github.com/ercanburak/EVREAL), revision `398f03551b7f`.
The original MIT license is preserved in `LICENSE` in this directory.

Only the E2VID+ FlowNet reconstruction architecture and its dependencies are
retained. Unused color wrappers and other teacher implementations have been
removed, together with the HyperE2VID dynamic decoder path. Dynamic decoder
configurations are explicitly rejected. The E2VID+ static decoder, parameter
names, and recurrent state behavior are retained.
The official checkpoint's prediction head, including its two flow channels, is
preserved for exact weight loading; distillation consumes only `image`.

E2VID+ is from Stoffregen et al., *Reducing the Sim-to-Real Gap for Event
Cameras*, ECCV 2020. Please cite the original work when using this teacher.
Its pretrained weights are distributed separately through
[EVREAL](https://github.com/ercanburak/EVREAL/tree/main/pretrained/E2VID%2B).
