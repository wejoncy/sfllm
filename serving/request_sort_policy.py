
class RequestSortPolicy:
    """
    RequestSortPolicy is a class that implements a sorting policy for requests.
    It sorts the requests based on their arrival time in ascending order.
    """

    def __init__(self):
        pass

    def sort_requests(self, requests)-> dict:
        """
        Sorts the given list of requests based on their token length.
        The purpose of this sorting is to optimize the processing time
        by prioritizing the same length requests.

        :param requests: List of requests to be sorted
        :return: Sorted list of requests
        """
        requests = sorted(requests, key=lambda x: len(x[-1].input_ids[0]))
        request_groups = {}
        for request in requests:
            # Group requests by their token length
            if len(request[-1].input_ids[0]) not in request_groups:
                request_groups[len(request[-1].input_ids[0])] = []
            # Append the request to the corresponding group
            request_groups[len(request[-1].input_ids[0])].append(request)
        return request_groups