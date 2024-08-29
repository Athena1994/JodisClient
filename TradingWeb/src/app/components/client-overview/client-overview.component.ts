import { Component, OnInit } from '@angular/core';
import { Client } from '../../models/client.interface';
import { ClientService } from '../../services/clients.service';
import { CommonModule, NgFor, NgIf } from '@angular/common';
import { MatListModule } from '@angular/material/list';
import { select, Store } from '@ngrx/store';
import { selectClients } from '../../state/clients/clients.selectors';
import { clientActions } from '../../state/clients/clients.actions';


@Component({
  selector: 'app-client-overview',
  standalone: true,
  imports: [CommonModule, MatListModule],
  templateUrl: './client-overview.component.html',
  styleUrl: './client-overview.component.css'
})

export class ClientOverviewComponent implements OnInit {
    clients$ = this.store.pipe(select(selectClients));

    constructor(private store: Store) {}

    ngOnInit() {
        this.store.dispatch(clientActions.load());
    }

}
