import { Component, OnInit } from '@angular/core';
import { Client } from '../../models/client.interface';
import { ClientService } from '../../services/clients.service';
import { NgFor, NgIf } from '@angular/common';
import { MatListModule } from '@angular/material/list';


@Component({
  selector: 'app-client-overview',
  standalone: true,
  imports: [NgFor, NgIf, MatListModule],
  templateUrl: './client-overview.component.html',
  styleUrl: './client-overview.component.css'
})

export class ClientOverviewComponent implements OnInit {
    clients: Client[] = [];

    constructor(private clientService: ClientService) {}

    ngOnInit() {
        this.clientService.getClients().subscribe((clients) => {
            this.clients = clients;
        })
    }

}
