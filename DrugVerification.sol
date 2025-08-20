// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DrugVerification {
    struct Batch {
        string productName;
        bool isLegitimate;
    }
    
    mapping(string => Batch) public batches;

    function addBatch(string memory _batchId, string memory _productName, bool _isLegitimate) public {
        batches[_batchId] = Batch(_productName, _isLegitimate);
    }
    
    function verifyBatch(string memory _batchId) public view returns (string memory, bool) {
        Batch memory batch = batches[_batchId];
        return (batch.productName, batch.isLegitimate);
    }
}
