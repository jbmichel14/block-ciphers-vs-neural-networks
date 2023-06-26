# Attacking Pseudorandomness With Deep Learning

Semester Project (Spring 2023) at [LASEC](https://lasec.epfl.ch) Lab (EPFL)

- Student: Jean-Baptiste Michel
- Supervisor: Ritam Bhaumik

## Description

Pseudorandomness generators are a central component of modern cryptosystems, and they typically rely on the fact that the patterns they leave on their outputs are hidden enough to fool conventional tests of randomness. With the advent of deep learning, it becomes interesting to ask whether deep neural networks can detect the patterns in the outputs of pseudorandomness generators. In this project, we will primarily be interested in a specific class of pseudorandomness generators: symmetric key modes of operation, which take a source of randomness and use it to build a function whose outputs look random. Deep learning approaches to attack such modes could range from brute-force classification algorithms (real vs random) to using more sophisticated cryptanalytic techniques but improving them through deep learning (for instance, in certain search problems). The goal of this project will be to explore the effectiveness of deep learning based cryptanalysis techniques against various modes of encryption, authentication, hashing, and so on, and develop some deep-learning based tests of effectiveness for pseudorandomness generators.

## Structure

| File/Directory&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Explanation                                                                                                                                                  |
|----------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| notes | This directory contains several text files with daily notes, notes about papers read during the project and some results of the experiments. |
| src | This directory contains the source code for the attack framework as well as Jupyter Notebooks for all experiments conducted during the project. |

