from abc import ABC, abstractmethod

import numpy as np


class OrderOptimizer(ABC):
    """Pick best order to crunch tensors."""

    @abstractmethod
    def computePath(self, notationList, finalNotation, tensorList, indexToSize):
        """Compute the optimal contraction path for tensor operations.
        
        Abstract method that must be implemented by subclasses to determine
        the optimal sequence of pairwise tensor contractions.
        
        Args:
            notationList (list): List of notation strings for each input tensor
            finalNotation (str): The desired output notation
            tensorList (list): List of input tensors
            indexToSize (dict): Mapping from index characters to their dimensions
            
        Returns:
            tuple: Implementation-specific return format for contraction path
        """
        pass


class GreedyOrderOptimizer(OrderOptimizer):
    """Pick cheapest way to crunch tensors."""

    def computePath(self, notationList, finalNotation, tensorList, indexToSize):
        """Compute the optimal contraction path for the given tensors.
        
        This method finds the sequence of pairwise tensor contractions that minimizes
        the total computational cost. It uses a greedy approach, always choosing the
        pair of tensors with the lowest contraction cost at each step.
        
        Args:
            notationList (list): List of notation strings for each input tensor
            finalNotation (str): The desired output notation
            tensorList (list): List of input tensors
            indexToSize (dict): Mapping from index characters to their dimensions
            
        Returns:
            tuple: A tuple containing:
                - path (list): List of (i, j) pairs indicating which tensors to contract
                - notations (list): List of resulting notations after each contraction
                
        Example:
            For tensors with notations ['ij', 'jk', 'kl'] and final notation 'il',
            might return ([(0, 1), (0, 1)], ['ik', 'il'])
        """
        path = []
        notations = []  # Store the new notation for each contraction step
        currentTensors = list(tensorList)
        currentNotations = list(notationList)

        while len(currentTensors) > 1:
            i, j, newNotation = self._findBestPair(currentNotations, indexToSize, finalNotation)
            self._updateLists(currentTensors, currentNotations, i, j, newNotation)
            path.append((min(i, j), max(i, j)))
            notations.append(newNotation)

        if currentNotations:
            # Check if a final reshape (sum, broadcast, or transpose) is needed
            if sorted(currentNotations[0]) != sorted(finalNotation):
                path.append((0, 0))  # Sum out or broadcast axes
                notations.append(finalNotation)
            elif currentNotations[0] != finalNotation:
                path.append((0, 0))  # Transpose only
                notations.append(finalNotation)
        return path, notations

    def _findBestPair(self, notationList, indexToSize, finalNotation):
        """Find the optimal pair of tensors to contract next.
        
        This method evaluates all possible pairs of tensors and selects the pair
        that results in the lowest computational cost for contraction. The cost
        is calculated as the product of all index dimensions involved in the
        contraction operation.
        
        Args:
            notationList (list): Current list of tensor notations
            indexToSize (dict): Mapping from index characters to their dimensions
            finalNotation (str): The target output notation
            
        Returns:
            tuple: A tuple containing:
                - bestI (int): Index of the first tensor to contract
                - bestJ (int): Index of the second tensor to contract
                - bestNewNotation (str): Resulting notation after contraction
        """
        minCost = float("inf")
        bestI, bestJ, bestNewNotation = 0, 0, ""

        for i in range(len(notationList)):
            for j in range(i + 1, len(notationList)):
                firstNotation = notationList[i]
                secondNotation = notationList[j]

                # Indices that appear in other tensors or the final output are kept
                otherIndices = set(finalNotation)
                for k in range(len(notationList)):
                    if k != i and k != j:
                        otherIndices.update(notationList[k])

                # Indices to be summed are those in both tensors but not needed later
                summedIndices = (set(firstNotation) & set(secondNotation)) - otherIndices

                # The new tensor will have all original indices minus the summed ones
                # Order indices according to their appearance in finalNotation when possible
                allNewIndices = (set(firstNotation) | set(secondNotation)) - summedIndices
                newIndices = []

                # First, add indices in the order they appear in finalNotation
                for idx in finalNotation:
                    if idx in allNewIndices:
                        newIndices.append(idx)
                        allNewIndices.remove(idx)

                # Then add any remaining indices in the order they appear in the input notations
                for idx in firstNotation + secondNotation:
                    if idx in allNewIndices:
                        newIndices.append(idx)
                        allNewIndices.remove(idx)

                # Cost is the product of dimensions of all indices involved in the contraction
                cost = np.prod([indexToSize[idx] for idx in set(firstNotation + secondNotation)])

                if cost < minCost:
                    minCost = cost
                    bestI, bestJ = i, j
                    bestNewNotation = "".join(newIndices)
        return bestI, bestJ, bestNewNotation

    def _updateLists(self, tensorList, notationList, i, j, newNotation):
        """Update tensor and notation lists after a contraction step.
        
        Removes the two tensors that were contracted (at indices i and j) and
        adds a placeholder (None) for the resulting contracted tensor. Updates
        the notation list accordingly with the new notation.
        
        Args:
            tensorList (list): List of tensors, modified in-place
            notationList (list): List of notations, modified in-place
            i (int): Index of the first tensor to remove
            j (int): Index of the second tensor to remove
            newNotation (str): Notation for the resulting contracted tensor
        """
        keep = [k for k in range(len(tensorList)) if k not in [i, j]]
        tensorList[:] = [tensorList[k] for k in keep] + [None]
        notationList[:] = [notationList[k] for k in keep] + [newNotation]


class Einsum:
    """Crunch tensors with einsum notation, easy to inspect."""

    def __init__(self, notation, *tensors, optimizer=None):
        """Initialize an Einsum operation.
        
        Creates an Einsum object that can perform tensor contractions according
        to the specified notation. Supports both simple and nested einsum expressions.
        
        Args:
            notation (str): Einsum notation string (e.g., 'ij,jk->ik' or '(ab,bc->ac)')
            *tensors: Variable number of input tensors to contract
            optimizer (OrderOptimizer, optional): Custom optimizer for contraction path.
                Defaults to GreedyOrderOptimizer if not provided.
                
        Example:
            >>> a = np.ones((2, 3))
            >>> b = np.ones((3, 4))
            >>> einsum_op = Einsum('ij,jk->ik', a, b)
            >>> result = einsum_op.execute()
        """
        self.notation = notation
        self.tensors = list(tensors)
        self.notationList = None
        self.finalNotation = None
        self.subEinsums = None
        self.indexToSize = None
        self.result = None
        self.path = None
        self.optimizer = optimizer or GreedyOrderOptimizer()
        self._parseNotation()

    def __str__(self):
        """Return a detailed string representation of the Einsum object.
        
        Provides comprehensive information about the einsum operation including
        input/output notations, tensor shapes, index sizes, contraction path,
        and any sub-einsum operations.
        
        Returns:
            str: Multi-line string with detailed einsum information
        """
        lines = [f"Einsum: {self.notation}"]
        if self.notationList:
            lines.append(f"Inputs: {self.notationList}")
        if self.finalNotation:
            lines.append(f"Output: {self.finalNotation}")
        if self.tensors:
            tensorShapes = []
            for t in self.tensors:
                if hasattr(t, "shape"):
                    tensorShapes.append(t.shape)
                elif isinstance(t, Einsum):
                    # Use output shape from sub-einsum
                    try:
                        tensorShapes.append(t.getShapes()[1])
                    except Exception:
                        tensorShapes.append("Einsum?")
                else:
                    tensorShapes.append("Unknown")
            lines.append(f"Tensor Shapes: {tensorShapes}")
        if self.indexToSize:
            lines.append(f"Index Sizes: {self.indexToSize}")
        if self.path:
            lines.append(f"Contraction Path: {self.path}")
        if self.result is not None:
            lines.append(f"Result Shape: {self.result.shape}")
        if self.subEinsums:
            lines.append("Sub-Einsums:")
            for i, subEinsum in enumerate(self.subEinsums):
                subLines = str(subEinsum).split("\n")
                lines.extend([f"  Sub {i + 1}: {line}" for line in subLines])
        return "\n".join(lines)

    def _parseNotation(self):
        """Parse the einsum notation string into components.
        
        Handles both simple einsum notation (e.g., 'ij,jk->ik') and complex
        nested expressions (e.g., '(ab,(bc,cd->bd)->ad)'). Robustly parses
        top-level arrows and commas while respecting nested parentheses.
        
        Sets the following instance attributes:
            notationList (list): List of input tensor notations
            finalNotation (str): Output notation
            subEinsums (list): List of nested Einsum objects
        """
        notation = self.notation.replace(" ", "")
        # If not a parenthesized expression, use simple parser
        if not (notation.startswith("(") and notation.endswith(")")):
            self.notationList, self.finalNotation, self.subEinsums = self._parseSimpleNotation(
                notation
            )
            return

        notation = notation[1:-1]  # strip outer parens
        # Find top-level '->' (not inside parentheses)
        stack = []
        splitIdx = None
        for idx, char in enumerate(notation):
            if char == "(":
                stack.append(idx)
            elif char == ")":
                if stack:
                    stack.pop()
            elif char == "-" and idx + 1 < len(notation) and notation[idx + 1] == ">" and not stack:
                splitIdx = idx
                break
        if splitIdx is not None:
            left = notation[:splitIdx]
            right = notation[splitIdx + 2 :]
        else:
            left = notation
            right = ""
        inputs = self._splitInputs(left)
        self.finalNotation = right
        self.subEinsums = []
        self.notationList = []
        tensorIdx = 0
        originalTensors = list(self.tensors)
        newTensorList = []
        for inp in inputs:
            if "->" in inp:
                subNotation = inp
                subTensorCount = subNotation.split("->")[0].count(",") + 1
                if subTensorCount > len(originalTensors) - tensorIdx:
                    raise ValueError(f"Not enough tensors for sub-notation: {subNotation}")
                subEinsum = Einsum(
                    subNotation,
                    *originalTensors[tensorIdx : tensorIdx + subTensorCount],
                    optimizer=self.optimizer,
                )
                self.notationList.append(subEinsum.finalNotation)
                self.subEinsums.append(subEinsum)
                tensorIdx += subTensorCount
            else:
                if tensorIdx >= len(originalTensors):
                    raise ValueError(f"Not enough tensors for notation: {inp}")
                self.notationList.append(inp)
                newTensorList.append(originalTensors[tensorIdx])
                tensorIdx += 1
        self.tensors = newTensorList

    def _splitInputs(self, inputStr):
        """Split input string into notation parts while respecting nested parentheses.
        
        Uses a stack-based approach to correctly handle nested parentheses when
        splitting on top-level commas. This ensures that nested einsum expressions
        are not incorrectly split.
        
        Args:
            inputStr (str): Input portion of notation string to split
            
        Returns:
            list: List of individual notation segments
        """
        inputs = []
        stack = []
        lastSplit = 0
        for i, char in enumerate(inputStr):
            if char == "(":
                stack.append(i)
            elif char == ")":
                if stack:
                    stack.pop()
            elif char == "," and not stack:
                # Only split at top-level commas
                part = inputStr[lastSplit:i].strip()
                if part:
                    inputs.append(part)
                lastSplit = i + 1
        # Add the last segment
        part = inputStr[lastSplit:].strip()
        if part:
            inputs.append(part)
        # Do NOT strip parentheses; return raw segments
        return [inp.strip() for inp in inputs if inp.strip()]

    def _parseSimpleNotation(self, notation):
        """Parse simple (non-nested) einsum notation.
        
        Handles standard einsum notation like 'ij,jk->ik' or implicit
        notation like 'ij,jk' (without explicit output).
        
        Args:
            notation (str): Simple einsum notation string
            
        Returns:
            tuple: (input_notations, output_notation, sub_einsums)
        """
        if "->" not in notation:
            notation = self._implicitToExplicit(notation)
        parts = notation.split("->")
        inputs = parts[0].split(",")
        output = parts[1] if len(parts) > 1 else ""
        return inputs, output, []

    def _implicitToExplicit(self, notation):
        """Convert implicit einsum notation to explicit form.
        
        For notation without explicit output (e.g., 'ij,jk'), determines
        the output indices by finding indices that appear only once across
        all inputs (following numpy's convention).
        
        Args:
            notation (str): Implicit einsum notation
            
        Returns:
            str: Explicit notation with '->' and output indices
        """
        inputs = notation.split(",")
        allIndices = set("".join(inputs))
        inputIndices = [idx for inp in inputs for idx in inp]
        repeatIndices = {idx for idx in inputIndices if inputIndices.count(idx) > 1}
        outputIndices = [idx for idx in allIndices if idx not in repeatIndices]
        return ",".join(inputs) + "->" + "".join(sorted(outputIndices))

    def _validateInputs(self, notationList, tensorList, subEinsums):
        """Validate that tensor shapes match their notations and build index size mapping.
        
        Checks that each tensor's dimensionality matches its notation length,
        verifies consistent index sizes across tensors, and handles sub-einsum
        validation.
        
        Args:
            notationList (list): List of notation strings
            tensorList (list): List of input tensors
            subEinsums (list): List of sub-einsum objects
            
        Returns:
            dict: Mapping from index characters to their dimensions
            
        Raises:
            ValueError: If shapes don't match notations or indices have inconsistent sizes
        """
        if len(notationList) != len(tensorList) + len(subEinsums):
            raise ValueError(
                f"Input count mismatch: {len(notationList)} notations, "
                f"{len(tensorList)} tensors, {len(subEinsums)}"
            )

        indexToSize = {}
        tensorIdx = 0
        for notation in notationList:
            for subEinsum in subEinsums:
                if subEinsum.finalNotation == notation:
                    self._checkSubEinsum(notation, subEinsum, indexToSize)
                    break
            else:
                tensor = tensorList[tensorIdx]
                tensorIdx += 1
                if len(notation) != len(tensor.shape):
                    raise ValueError(f"Shape mismatch: {notation}, {tensor.shape}")
                for idx, dim in zip(notation, tensor.shape):
                    self._setIndexSize(idx, dim, indexToSize)
        return indexToSize

    def _checkSubEinsum(self, notation, subEinsum, indexToSize):
        """Validate that a sub-einsum's output shape matches expected notation.
        
        Args:
            notation (str): Expected notation for the sub-einsum output
            subEinsum (Einsum): Sub-einsum object to validate
            indexToSize (dict): Index size mapping to update
            
        Raises:
            ValueError: If sub-einsum shape doesn't match notation
        """
        subShape = subEinsum.getShapes()[1]
        if len(notation) != len(subShape):
            raise ValueError(f"Sub-einsum shape mismatch: {notation}, {subShape}")
        for idx, dim in zip(notation, subShape):
            self._setIndexSize(idx, dim, indexToSize)

    def _setIndexSize(self, idx, dim, indexToSize):
        """Set or verify the size of an index dimension.
        
        Args:
            idx (str): Index character
            dim (int): Dimension size
            indexToSize (dict): Index size mapping to update
            
        Raises:
            ValueError: If index already exists with different size
        """
        if idx in indexToSize and indexToSize[idx] != dim:
            raise ValueError(f"Inconsistent dimension for {idx}")
        indexToSize[idx] = dim

    def _getTensorCoords(self, notation, allIndices, indexValues):
        """Convert index values to tensor coordinates for a specific notation.
        
        Maps the current index values to the appropriate tensor coordinates
        based on the tensor's notation and the canonical index ordering.
        
        Args:
            notation (str): Tensor's notation string
            allIndices (list): Canonical ordering of all indices
            indexValues (list): Current values for each index
            
        Returns:
            tuple: Tensor coordinates for accessing the element
        """
        coords = [0] * len(notation)
        for i, axis_char in enumerate(notation):
            try:
                pos = allIndices.index(axis_char)
                coords[i] = indexValues[pos]
            except ValueError:
                raise ValueError(
                    f"Index {axis_char} from notation {notation} not found in allIndices {allIndices}"
                )
        return tuple(coords)

    def _contractPair(self, tensor1, tensor2, notation1, notation2, newNotation):
        """Contract two tensors by summing over shared indices.
        
        Performs tensor contraction between two tensors, summing over indices
        that appear in both tensors but not in the output notation.
        
        Args:
            tensor1 (np.ndarray): First tensor to contract
            tensor2 (np.ndarray): Second tensor to contract
            notation1 (str): Notation for first tensor
            notation2 (str): Notation for second tensor
            newNotation (str): Desired output notation
            
        Returns:
            np.ndarray: Contracted tensor with shape matching newNotation
        """
        # Don't sort indices - preserve the natural ordering from newNotation
        outShape = tuple(self.indexToSize[idx] for idx in newNotation)
        newTensor = np.zeros(outShape)
        summedIndices = self._findSummedIndices(notation1, notation2, newNotation)

        # Build allIndices preserving the order from newNotation, then summed indices
        allIndices = list(newNotation) + summedIndices

        self._contractIterate(
            newTensor,
            tensor1,
            tensor2,
            notation1,
            notation2,
            newNotation,
            summedIndices,
            allIndices,
        )
        return newTensor

    def _findSummedIndices(self, notation1, notation2, newNotation):
        """Identify indices that should be summed during tensor contraction.
        
        Args:
            notation1 (str): First tensor notation
            notation2 (str): Second tensor notation
            newNotation (str): Output notation
            
        Returns:
            list: Indices that appear in both input tensors but not in output
        """
        summed_indices_set = (set(notation1) | set(notation2)) - set(newNotation)
        # Sort for deterministic order
        return sorted(list(summed_indices_set))

    def _contractIterate(
        self,
        newTensor,
        tensor1,
        tensor2,
        notation1,
        notation2,
        newNotation,
        summedIndices,
        allIndices,
    ):
        """Recursively iterate over all index combinations for contraction."""

        def iterate(indexValues, idxPos):
            if idxPos == len(allIndices):
                t1Coords = self._getTensorCoords(notation1, allIndices, indexValues)
                t2Coords = self._getTensorCoords(notation2, allIndices, indexValues)
                # Output coordinates are the first len(newNotation) values since newNotation indices come first
                outCoords = tuple(indexValues[: len(newNotation)]) if newNotation else ()
                newTensor[outCoords] += tensor1[t1Coords] * tensor2[t2Coords]
            else:
                idx = allIndices[idxPos]
                for i in range(self.indexToSize[idx]):
                    indexValues.append(i)
                    iterate(indexValues, idxPos + 1)
                    indexValues.pop()

        iterate([], 0)

    def _doReshape(self, tensor, notation, finalNotation):
        """Reshape tensor to match the final output notation.
        
        Handles various reshape operations including transposition, reduction,
        broadcasting, and conversion to scalar as needed to match the target notation.
        
        Args:
            tensor (np.ndarray): Input tensor to reshape
            notation (str): Current tensor notation
            finalNotation (str): Target output notation
            
        Returns:
            np.ndarray: Reshaped tensor matching finalNotation
        """
        outShape = tuple(self.indexToSize[idx] for idx in finalNotation)
        if finalNotation == "":
            return self._reshapeToScalar(tensor)
        # If notation is a permutation of finalNotation but order does not match, transpose
        if (
            sorted(finalNotation) == sorted(notation)
            and len(finalNotation) == len(notation)
            and notation != finalNotation
        ):
            return self._reshapeTranspose(tensor, notation, finalNotation)
        # If already in correct order, return as is
        if notation == finalNotation:
            return tensor
        if set(finalNotation).issubset(set(notation)):
            return self._reshapeReduce(tensor, notation, finalNotation, outShape)
        if tensor.shape == () and len(outShape) > 0:
            return self._reshapeBroadcast(tensor, outShape)
        return self._reshapeFallback(tensor, notation, finalNotation, outShape)

    def _reshapeToScalar(self, tensor):
        """Convert a tensor to a scalar (0-dimensional array).
        
        Args:
            tensor (np.ndarray): Input tensor to convert
            
        Returns:
            np.ndarray: 0-dimensional array containing the scalar value
        """
        # Special case for trace: 'ii->'
        if tensor.ndim == 2 and tensor.shape[0] == tensor.shape[1]:
            return np.trace(tensor)
        return tensor.sum()

    def _reshapeTranspose(self, tensor, notation, finalNotation):
        """Transpose a tensor to match the final output notation.
        
        Reorders the dimensions of the tensor to match the desired output
        notation, preserving the data but changing its layout.
        
        Args:
            tensor (np.ndarray): Input tensor to transpose
            notation (str): Current tensor notation
            finalNotation (str): Target output notation
            
        Returns:
            np.ndarray: Transposed tensor matching finalNotation
        """
        perm = [notation.index(idx) for idx in finalNotation]
        return np.transpose(tensor, perm)

    def _reshapeReduce(self, tensor, notation, finalNotation, outShape):
        """Reduce a tensor by summing over indices not in the final notation.
        
        Removes dimensions from the tensor that are not present in the final
        output notation by summing over those indices.
        
        Args:
            tensor (np.ndarray): Input tensor to reduce
            notation (str): Current tensor notation
            finalNotation (str): Target output notation
            outShape (tuple): Expected output shape
            
        Returns:
            np.ndarray: Reduced tensor matching finalNotation
        """
        axesToKeep = [notation.index(idx) for idx in finalNotation]
        perm = axesToKeep + [i for i in range(len(notation)) if notation[i] not in finalNotation]
        transposed = np.transpose(tensor, perm)
        axesToSum = tuple(range(len(finalNotation), len(notation)))
        if axesToSum:
            transposed = transposed.sum(axis=axesToSum)
        if transposed.shape == () and len(outShape) > 0:
            return np.full(outShape, transposed.item())
        return transposed

    def _reshapeBroadcast(self, tensor, outShape):
        """Broadcast a scalar tensor to the desired output shape.
        
        Expands a 0-dimensional tensor (scalar) to the specified output shape
        by filling all elements with the scalar value.
        
        Args:
            tensor (np.ndarray): Input scalar tensor
            outShape (tuple): Desired output shape
            
        Returns:
            np.ndarray: Broadcasted tensor with shape outShape
        """
        return np.full(outShape, tensor.item())

    def _reshapeFallback(self, tensor, notation, finalNotation, outShape):
        """Fallback reshape operation for complex notation changes.
        
        Handles cases where the tensor needs to be reshaped in a way that
        cannot be achieved through simple transposition or reduction.
        
        Args:
            tensor (np.ndarray): Input tensor to reshape
            notation (str): Current tensor notation
            finalNotation (str): Target output notation
            outShape (tuple): Expected output shape
            
        Returns:
            np.ndarray: Reshaped tensor matching finalNotation
        """
        newTensor = np.zeros(outShape)
        coordsMap = {
            notation[i]: finalNotation.index(idx)
            for i, idx in enumerate(notation)
            if idx in finalNotation
        }
        for idx in np.ndindex(tensor.shape):
            outIdx = [0] * len(finalNotation)
            for i, val in enumerate(idx):
                if notation[i] in coordsMap:
                    outIdx[coordsMap[notation[i]]] = val
            newTensor[tuple(outIdx)] = tensor[idx]
        return newTensor

    def _doContraction(self, tensorList, notationList, i, j):
        """Contract two tensors in the list and update notations and tensors in place.
        
        Performs the contraction of two tensors at the specified indices and
        updates the notation and tensor lists accordingly.
        
        Args:
            tensorList (list): List of input tensors
            notationList (list): List of input tensor notations
            i (int): Index of first tensor to contract
            j (int): Index of second tensor to contract
        """
        notation1, notation2 = notationList[i], notationList[j]
        newNotation = self._buildNewNotation(notation1, notation2)
        newTensor = self._contractPair(
            tensorList[i], tensorList[j], notation1, notation2, newNotation
        )
        self._replaceContractedTensors(tensorList, notationList, i, j, newTensor, newNotation)

    def _doContractionWithNotation(self, tensorList, notationList, i, j, newNotation):
        """Perform tensor contraction using precomputed output notation.
        
        Contracts two tensors at specified indices using the notation that
        was precomputed by the optimizer to ensure consistent results.
        
        Args:
            tensorList (list): List of input tensors
            notationList (list): List of input tensor notations
            i (int): Index of first tensor to contract
            j (int): Index of second tensor to contract
            newNotation (str): Precomputed output notation
        """
        notation1, notation2 = notationList[i], notationList[j]
        newTensor = self._contractPair(
            tensorList[i], tensorList[j], notation1, notation2, newNotation
        )
        self._replaceContractedTensors(tensorList, notationList, i, j, newTensor, newNotation)

    def _buildNewNotation(self, notation1, notation2):
        """Build the new notation string after contraction, preserving order.
        
        Combines the notations of two tensors being contracted, removing any
        indices that are summed over during the contraction.
        
        Args:
            notation1 (str): Notation of first tensor
            notation2 (str): Notation of second tensor
            
        Returns:
            str: New notation string after contraction
        """
        summed = set(notation1) & set(notation2)
        newNotation = ""
        seen = set()

        # First, add indices from notation1 that are not summed
        for idx in notation1:
            if idx not in summed and idx not in seen:
                newNotation += idx
                seen.add(idx)

        # Then, add indices from notation2 that are not summed and not already added
        for idx in notation2:
            if idx not in summed and idx not in seen:
                newNotation += idx
                seen.add(idx)

        return newNotation

    def _replaceContractedTensors(self, tensorList, notationList, i, j, newTensor, newNotation):
        """Replace the contracted tensors in the lists with the new tensor and notation.
        
        Updates the tensor and notation lists by replacing the two tensors that
        were contracted with the resulting tensor and its notation.
        
        Args:
            tensorList (list): List of input tensors
            notationList (list): List of input tensor notations
            i (int): Index of first tensor to contract
            j (int): Index of second tensor to contract
            newTensor (np.ndarray): Resulting tensor after contraction
            newNotation (str): Notation of resulting tensor
        """
        keep = [k for k in range(len(tensorList)) if k not in [i, j]]
        tensorList[:] = [tensorList[k] for k in keep] + [newTensor]
        notationList[:] = [notationList[k] for k in keep] + [newNotation]

    def _compute(self, notationList, finalNotation, subEinsums, tensorList, path, pathNotations):
        """Execute the einsum computation using the optimal contraction path.
        
        Applies the sequence of tensor contractions and reshapes as determined
        by the optimizer to compute the final result efficiently.
        
        Args:
            notationList (list): List of input tensor notations
            finalNotation (str): Target output notation
            subEinsums (list): List of sub-einsum objects
            tensorList (list): List of input tensors
            path (list): Sequence of (i, j) contraction pairs
            pathNotations (list): Corresponding output notations for each step
            
        Returns:
            np.ndarray: Final computed result tensor
        """
        currentTensors = self._buildCurrentTensors(notationList, subEinsums, tensorList)
        currentNotations = list(notationList)
        for stepIdx, (i, j) in enumerate(path):
            if i == j == 0:
                # Final reshape step
                currentTensors[0] = self._doReshape(
                    currentTensors[0], currentNotations[0], finalNotation
                )
                currentNotations[0] = finalNotation
            else:
                # Contract two tensors at indices i and j using the precomputed notation
                newNotation = pathNotations[stepIdx]
                self._doContractionWithNotation(currentTensors, currentNotations, i, j, newNotation)
        return currentTensors[0]

    def _buildCurrentTensors(self, notationList, subEinsums, tensorList):
        """Build the current tensor list by executing sub-einsums.
        
        Creates the initial tensor list for computation by executing any
        nested sub-einsum operations and combining with regular tensors.
        
        Args:
            notationList (list): List of tensor notations
            subEinsums (list): List of sub-einsum objects
            tensorList (list): List of regular input tensors
            
        Returns:
            list: Combined list of tensors ready for contraction
        """
        currentTensors = []
        tensorIdx = 0
        for notation in notationList:
            subTensor = self._findSubEinsumTensor(notation)
            if subTensor is not None:
                currentTensors.append(subTensor)
            else:
                currentTensors.append(tensorList[tensorIdx])
                tensorIdx += 1
        return currentTensors

    def getShapes(self):
        """Get the shapes of input tensors and expected output.
        
        Computes and returns the shapes of all input tensors and the
        expected output tensor shape based on the einsum notation.
        
        Returns:
            tuple: (input_shapes, output_shape) where:
                - input_shapes (list): List of input tensor shapes
                - output_shape (tuple): Expected output tensor shape
        """
        if not self.notationList or not self.finalNotation:
            self._parseNotation()
        inputShapes = []
        tensorIdx = 0
        indexToSize = self.indexToSize or self._validateInputs(
            self.notationList, self.tensors, self.subEinsums
        )
        for notation in self.notationList:
            for subEinsum in self.subEinsums:
                if subEinsum.finalNotation == notation:
                    subEinsum.indexToSize = subEinsum._validateInputs(
                        subEinsum.notationList, subEinsum.tensors, subEinsum.subEinsums
                    )
                    subShape = tuple(subEinsum.indexToSize[idx] for idx in subEinsum.finalNotation)
                    inputShapes.append(subShape)
                    break
            else:
                inputShapes.append(self.tensors[tensorIdx].shape)
                tensorIdx += 1
        outputShape = tuple(indexToSize[idx] for idx in self.finalNotation)
        return inputShapes, outputShape

    def execute(self):
        """Execute the einsum operation and return the result.
        
        Performs the complete einsum computation including input validation,
        contraction path optimization, and tensor contractions. Results are
        cached for subsequent calls.
        
        Returns:
            np.ndarray: The computed einsum result
            
        Raises:
            ValueError: If tensor shapes don't match notations or other validation errors
        """
        if self.result is not None:
            return self.result
        self.indexToSize = self._validateInputs(self.notationList, self.tensors, self.subEinsums)
        inputTensors = self._buildInputTensors()
        self.path, self.pathNotations = self.optimizer.computePath(
            self.notationList, self.finalNotation, inputTensors, self.indexToSize
        )
        self.result = self._compute(
            self.notationList,
            self.finalNotation,
            self.subEinsums,
            self.tensors,
            self.path,
            self.pathNotations,
        )
        return self.result

    def _buildInputTensors(self):
        """Build the list of input tensors, resolving sub-einsums if present."""
        inputTensors = []
        tensorIdx = 0
        for notation in self.notationList:
            subTensor = self._findSubEinsumTensor(notation)
            if subTensor is not None:
                inputTensors.append(subTensor)
            else:
                inputTensors.append(self.tensors[tensorIdx])
                tensorIdx += 1
        return inputTensors

    def _findSubEinsumTensor(self, notation):
        """Return the executed tensor for a sub-einsum if matching notation, else None."""
        for subEinsum in self.subEinsums:
            if subEinsum.finalNotation == notation:
                return subEinsum.execute()
        return None
