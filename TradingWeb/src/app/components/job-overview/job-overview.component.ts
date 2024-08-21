import { Component } from '@angular/core';
import { MatListModule } from '@angular/material/list';
import { Job } from '../../models/job.interface';
import { JobService } from '../../services/jobs.service';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-job-overview',
  standalone: true,
  imports: [MatListModule, CommonModule],
  templateUrl: './job-overview.component.html',
  styleUrl: './job-overview.component.css'
})

export class JobOverviewComponent {
    assignedJobs: Job[] = [];
    unassignedJobs: Job[] = [];
    finishedJobs: Job[] = [];

    constructor(private jobService: JobService) {}

    ngOnInit() {
      this.jobService.getJobs(true, false, false).subscribe((jobs) => {
          this.assignedJobs = jobs;
      });
      this.jobService.getJobs(false, true, false).subscribe((jobs) => {
          this.unassignedJobs = jobs;
      });
      this.jobService.getJobs(false, false, true).subscribe((jobs) => {
          this.finishedJobs = jobs;
      });
    }
}
