import asyncio


class Scheduler:
    def __init__(self, queue_manager):
        self.pending_requests = []
        self.running_requests = []
        self.queue_manager = queue_manager

    def schedule(self):
        # Get the next request with a timeout
        while True:
            request = self.queue_manager.get_next_request()
            if request is None or len(self.pending_requests) >= 10:
                self.running_requests.extend(self.pending_requests)
                self.pending_requests.clear()
                break
            self.pending_requests.append(request)
        
        return self.running_requests