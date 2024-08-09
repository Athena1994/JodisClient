import subprocess

from program.client.client import Client


def main():
    client = Client(1000)
    client.spawn_worker("examples/example_job.zip")
    print(client._working_dir.get())
    while True:
        try:
            client._worker_proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            print(client.get_job_state())
            pass
        else:
            break

    client.stop_worker()


if __name__ == '__main__':
    main()
