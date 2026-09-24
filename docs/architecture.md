# Architecture

## Overview

img_gen_01 is organized around a separation between the application code and the model weights.

~~~text
                    +----------------+
                    |   User / UI    |
                    +-------+--------+
                            |
                            v
                    +----------------+
                    | Prompt Handler |
                    +-------+--------+
                            |
                            v
                    +----------------+
                    | Model Interface|
                    +-------+--------+
                            |
                            v
                    +----------------+
                    | Generation     |
                    | Pipeline       |
                    +-------+--------+
                            |
                            v
                    +----------------+
                    | Image Decoder  |
                    +-------+--------+
                            |
                            v
                    +----------------+
                    | Output Image   |
                    +----------------+
~~~

## Design principles

1. Local-first — inference should work without a paid hosted generation API.
2. Modular — each major component should have a clear responsibility.
3. Replaceable models — the application should not hard-code one model forever.
4. Inspectable — important steps should be understandable and testable.
5. Incremental development — begin with a working small system and expand it.

## Model boundary

The model is treated as a component with a defined interface. This makes it possible to replace one model with another without rewriting the whole application.

## Hardware

The application should eventually support hardware-aware execution so that memory-heavy operations can be configured according to the available GPU/CPU resources.
